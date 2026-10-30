import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pytorch_msssim


# =============================
# U-Net Model
# =============================
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        prev_channels = in_channels
        for feature in features:
            self.encoder_layers.append(UNetBlock(prev_channels, feature))
            prev_channels = feature

        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        prev_channels = features[-1] * 2
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(prev_channels, feature, kernel_size=2, stride=2))
            self.decoder_layers.append(UNetBlock(feature * 2, feature))
            prev_channels = feature

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        skip_connections = []

        for encoder in self.encoder_layers:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip_connection = skip_connections[i]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_layers[i](x)

        x = self.final_conv(x)
        return self.output_activation(x).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet()
model.to(device)

model.load_state_dict(torch.load("u-net-hyperparameter.pt", weights_only=True))

for case in [18, 50]:
    case_path = f"BASE_DIRECTORY/case{case}.npy"
    
    if not os.path.exists(case_path):
        continue

    input_np = np.load(case_path)
    input_tensor = torch.from_numpy(input_np).float().to(device)
    input_tensor = input_tensor.unsqueeze(0)
    

    output = model(input_tensor)
    output_np = output.detach().cpu().numpy()
    np.save(f"predictions/Hyperparameters/prediction{case}.npy", output_np)

