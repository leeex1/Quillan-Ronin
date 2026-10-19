
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricImageDecoder(nn.Module):
    """
    🖼️ Phase 7: Vectorized Image Decoder
    Converts latent state [B, D] into 256x256 image [B, 3, 256, 256]
    """
    def __init__(self, hidden_dim=1280):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 512 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 256x256
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, x):
        h = self.fc(x).view(-1, 512, 8, 8)
        return self.deconv(h)

class GeometricAudioDecoder(nn.Module):
    """
    🎵 Phase 7: Vectorized Audio Decoder
    Converts latent state [B, D] into 1-sec waveform [B, 1, 16000]
    """
    def __init__(self, hidden_dim=1280):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 256 * 125)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=4, padding=0),  # 500
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, stride=4, padding=0),   # 2000
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, stride=4, padding=0),   # 8000
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, 2, stride=2, padding=0),    # 16000
            nn.Tanh()
        )

    def forward(self, x):
        h = self.fc(x).view(-1, 256, 125)
        return self.deconv(h)

class GeometricVideoDecoder(nn.Module):
    """
    🎬 Phase 7: Vectorized Video Decoder
    Converts latent state [B, D] into 8-frame 64x64 video [B, 3, 8, 64, 64]
    """
    def __init__(self, hidden_dim=1280):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 256 * 2 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)), # 4x8x8
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, (2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),  # 8x16x16
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),   # 8x32x32
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 3, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),    # 8x64x64
            nn.Tanh()
        )

    def forward(self, x):
        h = self.fc(x).view(-1, 256, 2, 4, 4)
        return self.deconv(h)
