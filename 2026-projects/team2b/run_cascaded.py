import os
import torch
import numpy as np
import torch.nn as nn

# ==========================================
# ROI Extraction
# ==========================================

def extract_roi_mask(volume, roi_size=(24, 24, 24), threshold_frac=0.4):
    thresh = threshold_frac * volume.max()
    signal_mask = volume > thresh

    if signal_mask.sum() == 0:
        center = np.array(volume.shape) // 2
    else:
        coords = np.argwhere(signal_mask)
        center = coords.mean(axis=0).astype(int)

    half = np.array(roi_size) // 2
    lo = np.clip(center - half, 0, None)
    hi = np.clip(center + half, None, np.array(volume.shape))

    mask = np.zeros_like(volume, dtype=bool)
    mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return mask

# U-Net Building Block **************************************

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        return self.relu(out + residual)

# U-Net *************************************************

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
        if x.dim() == 4:
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

# ==========================================
# Main Inference
# ==========================================

def main():

# PATHS

    input_dir = "/path/data"

    tier1_weights = "tier1_best_weights.pt"

    tier2_weights = "tier2_best_weights.pt"

    output_dir = "predictions_new"

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------
    # DEVICE
    # ----------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ----------------------------------
    # LOAD TIER 1
    # ----------------------------------

    model_tier1 = UNet(
        in_channels=1,
        out_channels=1
    )

    model_tier1.load_state_dict(
        torch.load(
            tier1_weights,
            map_location=device,
            weights_only=True
        )
    )

    model_tier1.to(device)
    model_tier1.eval()

    print("Loaded Tier 1")

    # ----------------------------------
    # LOAD TIER 2
    # ----------------------------------

    model_tier2 = UNet(
        in_channels=2,
        out_channels=1
    )

    model_tier2.load_state_dict(
        torch.load(
            tier2_weights,
            map_location=device,
            weights_only=True
        )
    )

    model_tier2.to(device)
    model_tier2.eval()

    print("Loaded Tier 2")

    # ----------------------------------
    # FIND FILES
    # ----------------------------------

    filenames = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".npy")
    ])

    print(f"Found {len(filenames)} files")

    # ----------------------------------
    # INFERENCE LOOP
    # ----------------------------------

    with torch.no_grad():

        for fname in filenames:

            print(f"Processing {fname}")

            input_path = os.path.join(
                input_dir,
                fname
            )

            recon = np.load(input_path).astype(np.float32)

            # -----------------------
            # Tier 1
            # -----------------------

            recon_tensor = (
                torch.from_numpy(recon)
                .float()
                .unsqueeze(0)
                .to(device)
            )

            tier1_pred = model_tier1(recon_tensor)

            tier1_np = (
                tier1_pred
                .squeeze()
                .cpu()
                .numpy()
            )

            # -----------------------
            # ROI Mask
            # -----------------------

            mask = extract_roi_mask(tier1_np)

            masked_channel = recon * mask

            # -----------------------
            # Tier 2 Input
            # -----------------------

            tier2_input = np.stack(
                [masked_channel, recon],
                axis=0
            ).astype(np.float32)

            tier2_tensor = (
                torch.from_numpy(tier2_input)
                .float()
                .unsqueeze(0)
                .to(device)
            )

            # -----------------------
            # Tier 2 Prediction
            # -----------------------

            prediction = model_tier2(
                tier2_tensor
            )

            prediction_np = (
                prediction
                .squeeze()
                .cpu()
                .numpy()
            )

            save_path = os.path.join(
                output_dir,
                f"prediction_{fname}"
            )

            np.save(
                save_path,
                prediction_np
            )

            print(f"Saved {save_path}")

    print("Inference complete.")


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    main()
