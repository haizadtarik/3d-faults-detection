import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SeismicDataset(Dataset):
    def __init__(self, seismic_dir, fault_dir, dim):
        """
        Args:
            seismic_path: Directory containing seismic data files
            fault_path: Directory containing fault data files 
            filenames: List of filenames (without path)
            dim: Tuple of dimensions for reshaping
        """
        self.seismic_dir = seismic_dir
        self.fault_dir = fault_dir
        self.filenames = os.listdir(fault_dir)
        self.dim = dim
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load seismic and fault data using full paths
        seismic = np.fromfile(os.path.join(self.seismic_dir, filename), dtype=np.single)
        fault = np.fromfile(os.path.join(self.fault_dir, filename), dtype=np.single)
        
        # Reshape
        seismic = np.reshape(seismic, self.dim)
        fault = np.reshape(fault, self.dim)
        
        # Normalize seismic data
        xm = np.mean(seismic)
        xs = np.std(seismic)
        seismic = (seismic - xm) / xs
        
        # Transpose to match PyTorch format (C,D,H,W)
        seismic = np.expand_dims(seismic.transpose(), 0)
        fault = np.expand_dims(fault.transpose(), 0)
        
        # Convert to torch tensors
        return torch.FloatTensor(seismic), torch.FloatTensor(fault)

class SeismicData():
    def __init__(self):
        pass

    def load_dat(self, seismic_path, fault_path=None, dim=(128,128,128)):
        n1, n2, n3 = dim
        # Load and preprocess data
        seismic = np.fromfile(seismic_path, dtype=np.single)
        seismic = np.reshape(seismic, (n1,n2,n3))

        # Normalize
        gm = np.mean(seismic)
        gs = np.std(seismic)
        seismic = (seismic - gm) / gs

        # Transpose and prepare for PyTorch (C,D,H,W format)
        seismic = np.transpose(seismic)
        seismic = np.expand_dims(seismic, axis=(0,1))  # Add batch and channel dimensions

        if fault_path is not None:
            fault = np.fromfile(fault_path, dtype=np.single)
            fault = np.reshape(fault, (n1, n2, n3))
            fault = np.transpose(fault)
            fault = np.expand_dims(fault, axis=(0,1))
            return seismic, fault
        else:
            return seismic
    
    def load_dataset(self, seismic_path, fault_path, dim=(128,128,128), batch_size=4, num_workers=1):
        dataset = SeismicDataset(seismic_path, fault_path, dim)
        data_loader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        return data_loader