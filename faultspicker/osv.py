import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Optional
from models import UNet3D

class OptimalSurfaceVoter:
    """
    Optimal Surface Voting implementation for fault detection enhancement.
    Works with PyTorch U-Net predictions.
    """
    def __init__(self, 
                 ru: int = 10,  # window in fault normal direction
                 rv: int = 20,  # window in fault strike direction
                 rw: int = 30): # window in fault dip direction
        self.ru = ru
        self.rv = rv
        self.rw = rw
        self.strain_max1 = 0.25  # bound on strain in 1st dimension
        self.strain_max2 = 0.25  # bound on strain in 2nd dimension
        self.surface_smooth1 = 2.0  # smoothing in 1st dimension
        self.surface_smooth2 = 2.0  # smoothing in 2nd dimension
        
    def set_strain_max(self, strain1: float, strain2: float) -> None:
        """Set maximum strain bounds"""
        self.strain_max1 = strain1
        self.strain_max2 = strain2
        
    def set_surface_smoothing(self, smooth1: float, smooth2: float) -> None:
        """Set surface smoothing parameters"""
        self.surface_smooth1 = smooth1
        self.surface_smooth2 = smooth2

    def convert_unet_predictions(self, 
                               pred: torch.Tensor) -> np.ndarray:
        """Convert PyTorch U-Net predictions to numpy array"""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        return pred.squeeze()

    def pick_seeds(self, 
                  fault_prob: np.ndarray,
                  threshold: float = 0.3,
                  min_distance: int = 4) -> List[Tuple[int, int, int]]:
        """Pick seed points from fault probability volume"""
        seeds = []
        n3, n2, n1 = fault_prob.shape
        
        # Find local maxima above threshold
        for i3 in range(1, n3-1):
            for i2 in range(1, n2-1):
                for i1 in range(1, n1-1):
                    if fault_prob[i3,i2,i1] < threshold:
                        continue
                        
                    # Check if local maximum
                    window = fault_prob[i3-1:i3+2, i2-1:i2+2, i1-1:i1+2]
                    if fault_prob[i3,i2,i1] == np.max(window):
                        # Check minimum distance from existing seeds
                        too_close = False
                        for seed in seeds:
                            dist = np.sqrt((i3-seed[0])**2 + 
                                         (i2-seed[1])**2 + 
                                         (i1-seed[2])**2)
                            if dist < min_distance:
                                too_close = True
                                break
                        if not too_close:
                            seeds.append((i3,i2,i1))
        
        return seeds

    def compute_surface_normals(self, 
                              fault_prob: np.ndarray,
                              sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute fault surface normals using gradient"""
        # Smooth probability volume
        smoothed = gaussian_filter(fault_prob, sigma)
        
        # Compute gradients
        gy, gx, gz = np.gradient(smoothed)
        
        # Normalize gradients
        norm = np.sqrt(gx**2 + gy**2 + gz**2)
        norm[norm == 0] = 1.0
        
        return gx/norm, gy/norm, gz/norm

    def apply_voting(self,
                    fault_prob: np.ndarray,
                    min_distance: int = 4,
                    threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply optimal surface voting to enhance fault detection"""
        # Initialize voting score volume
        n3, n2, n1 = fault_prob.shape
        voting_scores = np.zeros_like(fault_prob)
        strike_volume = np.zeros_like(fault_prob)
        dip_volume = np.zeros_like(fault_prob)
        
        # Get seed points
        seeds = self.pick_seeds(fault_prob, threshold, min_distance)
        
        # Compute surface normals
        nx, ny, nz = self.compute_surface_normals(fault_prob)
        
        # For each seed point
        for i3, i2, i1 in seeds:
            # Extract local volume around seed
            r = self.rw
            local_vol = self._extract_local_volume(
                fault_prob, i3, i2, i1, r)
            
            # Find optimal surface in local volume
            surface = self._find_optimal_surface(local_vol)
            
            # Compute voting scores
            scores = self._compute_voting_scores(
                surface, local_vol)
            
            # Update voting volumes
            self._update_voting_volumes(
                voting_scores, strike_volume, dip_volume,
                scores, i3, i2, i1, r)
        
        return voting_scores, strike_volume, dip_volume

    def extract_surfaces(self,
                        voting_scores: np.ndarray,
                        strike_vol: np.ndarray,
                        dip_vol: np.ndarray,
                        threshold: float = 0.3) -> List[np.ndarray]:
        """Extract fault surfaces from voting results"""
        surfaces = []
        n3, n2, n1 = voting_scores.shape
        
        # Create binary mask of high voting scores
        mask = voting_scores > threshold
        
        # Label connected components
        from scipy.ndimage import label
        labels, num_features = label(mask)
        
        # Extract each surface
        for i in range(1, num_features + 1):
            surface_mask = labels == i
            if np.sum(surface_mask) > 100: # Min surface size
                surface = np.zeros((np.sum(surface_mask), 3))
                points = np.where(surface_mask)
                surface[:, 0] = points[0]
                surface[:, 1] = points[1] 
                surface[:, 2] = points[2]
                surfaces.append(surface)
                
        return surfaces

    def _extract_local_volume(self,
                            volume: np.ndarray,
                            i3: int,
                            i2: int,
                            i1: int,
                            r: int) -> np.ndarray:
        """Extract local volume around point"""
        n3, n2, n1 = volume.shape
        x1 = max(0, i1-r)
        x2 = min(n1, i1+r+1)
        y1 = max(0, i2-r)
        y2 = min(n2, i2+r+1)
        z1 = max(0, i3-r)
        z2 = min(n3, i3+r+1)
        return volume[z1:z2, y1:y2, x1:x2]

    def _find_optimal_surface(self,
                            local_vol: np.ndarray) -> np.ndarray:
        """
        Find optimal surface in local volume using dynamic programming
        """
        n3, n2, n1 = local_vol.shape
        cost = np.zeros_like(local_vol)
        cost[0] = local_vol[0]
        
        # Forward pass
        for i in range(1, n3):
            for j in range(n2):
                for k in range(n1):
                    j_min = max(0, j-1)
                    j_max = min(n2-1, j+1)
                    k_min = max(0, k-1)
                    k_max = min(n1-1, k+1)
                    cost[i,j,k] = local_vol[i,j,k] + np.max(
                        cost[i-1, j_min:j_max+1, k_min:k_max+1])
        
        # Backward pass to find surface
        surface = np.zeros((n3, 2), dtype=np.int32)
        i = n3-1
        surface[i] = np.unravel_index(
            np.argmax(cost[i]), (n2,n1))
        
        for i in range(n3-2, -1, -1):
            j, k = surface[i+1]
            j_min = max(0, j-1)
            j_max = min(n2-1, j+1)
            k_min = max(0, k-1)
            k_max = min(n1-1, k+1)
            idx = np.argmax(cost[i, j_min:j_max+1, k_min:k_max+1])
            surface[i] = [j_min + idx//(k_max-k_min+1),
                         k_min + idx%(k_max-k_min+1)]
        
        return surface

    def _compute_voting_scores(self,
                             surface: np.ndarray,
                             local_vol: np.ndarray) -> np.ndarray:
        """Compute voting scores for surface"""
        scores = np.zeros_like(local_vol)
        n3 = local_vol.shape[0]
        
        # Compute scores based on local maxima along surface
        for i in range(n3):
            j, k = surface[i]
            scores[i,j,k] = local_vol[i,j,k]
            
        # Smooth scores
        scores = gaussian_filter(scores, 
                               [self.surface_smooth1,
                                self.surface_smooth2,
                                self.surface_smooth2])
        
        return scores

    def _update_voting_volumes(self,
                             voting_scores: np.ndarray,
                             strike_vol: np.ndarray, 
                             dip_vol: np.ndarray,
                             scores: np.ndarray,
                             i3: int,
                             i2: int,
                             i1: int,
                             r: int) -> None:
        """Update voting volumes with local scores"""
        n3, n2, n1 = voting_scores.shape
        s3, s2, s1 = scores.shape
        
        x1 = max(0, i1-r)
        x2 = min(n1, i1+r+1)
        y1 = max(0, i2-r)
        y2 = min(n2, i2+r+1)
        z1 = max(0, i3-r)
        z2 = min(n3, i3+r+1)
        
        # Update voting scores
        voting_scores[z1:z2, y1:y2, x1:x2] = np.maximum(
            voting_scores[z1:z2, y1:y2, x1:x2],
            scores[:z2-z1, :y2-y1, :x2-x1])
        
    def visualize_surfaces(self, surfaces: List[np.ndarray], 
                        volume_shape: Tuple[int, int, int],
                        cmap: str = 'viridis') -> None:
        """
        Visualize extracted fault surfaces in 3D
        
        Args:
            surfaces: List of surface point arrays (N,3)
            volume_shape: Shape of original volume (depth,height,width)
            cmap: Matplotlib colormap name
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D figure
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each surface
        for surface in surfaces:
            scatter = ax.scatter(surface[:,2], # x coordinates
                            surface[:,1], # y coordinates  
                            surface[:,0], # z coordinates
                            c=surface[:,0], # color by depth
                            cmap=cmap,
                            alpha=0.6,
                            s=1)
        
        # Set labels and limits
        ax.set_xlabel('Width')
        ax.set_ylabel('Height') 
        ax.set_zlabel('Depth')
        ax.set_xlim(0, volume_shape[2])
        ax.set_ylim(0, volume_shape[1])
        ax.set_zlim(volume_shape[0], 0) # Reverse depth axis
        
        # Add colorbar
        plt.colorbar(scatter, label='Depth')
        plt.title('Extracted Fault Surfaces')
        plt.tight_layout()
        plt.show()
        
def enhance_fault_detection() -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Enhance fault detection from U-Net predictions using optimal surface voting
    
    Args:
        unet_pred: PyTorch tensor of U-Net predictions (N,C,D,H,W)
        
    Returns:
        enhanced_vol: Enhanced fault probability volume
        surfaces: List of extracted fault surfaces
    """
    model_path = '/home/harzad/projects/seismic-faults-picker/model/best_model.pth'
    seismic_path = '/home/harzad/projects/seismic-faults-picker/data/validation/fault/2.dat'
    n1,n2,n3 = 128,128,128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    seismic = np.fromfile(seismic_path, dtype=np.single)
    seismic = np.reshape(seismic, (n1,n2,n3))

    # Normalize
    gm = np.mean(seismic)
    gs = np.std(seismic)
    seismic = (seismic - gm) / gs

    # Transpose and prepare for PyTorch (C,D,H,W format)
    seismic = np.transpose(seismic)
    seismic = np.expand_dims(seismic, axis=(0,1))  # Add batch and channel dimensions
    seismic_data = torch.FloatTensor(seismic).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(seismic_data)
        prediction = prediction.cpu().numpy()[0,0]

    # Initialize voter
    voter = OptimalSurfaceVoter(ru=10, rv=20, rw=30)
    voter.set_strain_max(0.25, 0.25)
    voter.set_surface_smoothing(2.0, 2.0)
    
    # Convert predictions
    fault_prob = voter.convert_unet_predictions(prediction)
    
    # Apply voting
    voting_scores, strike_vol, dip_vol = voter.apply_voting(
        fault_prob, 
        min_distance=4,
        threshold=0.3
    )
    
    # Extract fault surfaces
    surfaces = voter.extract_surfaces(
        voting_scores,
        strike_vol,
        dip_vol,
        threshold=0.3
    )

    print('type(voting_scores):', type(voting_scores))
    print('voting_scores.shape:', voting_scores.shape)
    print('type(surfaces):', type(surfaces))
    print('len(surfaces):', len(surfaces))

    # visualize surface
    voter.visualize_surfaces(surfaces, fault_prob.shape)
    

if __name__ == '__main__':
    enhance_fault_detection()