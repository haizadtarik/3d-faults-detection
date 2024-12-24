import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from faultspicker.models import UNet3D

class FaultsPicker():
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = UNet3D()
        self.model.load_state_dict(torch.load(model_path))

    def train(self, train_loader, valid_loader, num_epochs=100, criterion=None, optimizer=None, save_path='model/best_model.pth', return_model=False):
        if self.model is None:
            self.model = UNet3D()
        if criterion is None:
            criterion = nn.BCELoss()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        best_valid_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    valid_loss += criterion(output, target).item()
            
            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(self.model.state_dict(), save_path)
        
        if return_model:
            return self.model

    def predict(self, seismic_data):
        if self.model is None:
            raise Exception('Model not loaded. Use load_model to load a model or run train to train a model.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seismic_data = torch.FloatTensor(seismic_data).to(device)
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(seismic_data)
            prediction = prediction.cpu().numpy()[0,0]
        return prediction
