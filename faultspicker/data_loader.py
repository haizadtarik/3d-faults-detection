import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SeismicDataset(Dataset):
    def __init__(self, seismic_path, fault_path, data_ids, dim):
        self.seismic_path = seismic_path
        self.fault_path = fault_path
        self.data_ids = data_ids
        self.dim = dim
        
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        
        # Load seismic and fault data
        seismic = np.fromfile(f"{self.seismic_path}{data_id}.dat", dtype=np.single)
        fault = np.fromfile(f"{self.fault_path}{data_id}.dat", dtype=np.single)
        
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
    def __init__(self, seismic_path, fault_path=None, n1=128, n2=128, n3=128):
        self.seismic_path = seismic_path
        self.fault_path = fault_path
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

    def load_dat(self):

        # Load and preprocess data
        seismic = np.fromfile(self.seismic_path, dtype=np.single)
        seismic = np.reshape(seismic, (self.n1,self.n2,self.n3))

        # Normalize
        gm = np.mean(seismic)
        gs = np.std(seismic)
        seismic = (seismic - gm) / gs

        # Transpose and prepare for PyTorch (C,D,H,W format)
        seismic = np.transpose(seismic)
        seismic = np.expand_dims(seismic, axis=(0,1))  # Add batch and channel dimensions

        if self.fault_path is not None:
            fault = np.fromfile(self.fault_path, dtype=np.single)
            fault = np.reshape(fault, (self.n1, self.n2, self.n3))
            fault = np.transpose(fault)
            fault = np.expand_dims(fault, axis=(0,1))
            return seismic, fault
        else:
            return seismic
    
    def load_dataset(self, seismic_path, fault_path, data_ids, batch_size=4, num_workers=1, dim=(128,128,128)):
        dataset = SeismicDataset(train_seismic_path, train_fault_path, data_ids, dim)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        return data_loader