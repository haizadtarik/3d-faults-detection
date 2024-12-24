import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(1, 16, 2)
        self.enc2 = self._make_layer(16, 32, 2)
        self.enc3 = self._make_layer(32, 64, 2)
        self.enc4 = self._make_layer(64, 128, 2)
        
        # Decoder
        self.dec1 = self._make_layer(128+64, 64, 2)
        self.dec2 = self._make_layer(64+32, 32, 2)
        self.dec3 = self._make_layer(32+16, 16, 2)
        
        # Final layer
        self.final = nn.Conv3d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Pool and upsample
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels, conv_num):
        layers = []
        for i in range(conv_num):
            if i == 0:
                layers.append(nn.Conv3d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        # Bridge
        e4 = self.enc4(p3)
        
        # Decoder
        d1 = self.dec1(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e1], dim=1))
        
        out = self.sigmoid(self.final(d3))
        return out
        