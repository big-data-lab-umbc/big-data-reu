import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import pytorch_msssim


def maybe_flip(a, b):
    for axis in [0, 1, 2]:
        if np.random.rand() < 0.5:
            a = np.flip(a, axis=axis).copy()
            b = np.flip(b, axis=axis).copy()
    return a, b


def random_intensity_scale(vol, low=0.9, high=1.1):
    scale = np.random.uniform(low, high)
    return vol * scale


def add_gaussian_noise(vol, std=0.015):
    noise = np.random.normal(0, std, vol.shape).astype(np.float32)
    return vol + noise


def gaussian_blur_3d(vol, sigma_range=(0.3, 0.7)):
    sigma = np.random.uniform(*sigma_range)
    return gaussian_filter(vol, sigma=sigma)


class PairedNPYDataset(Dataset):
    def __init__(self, input_dir, label_dir, filenames, mode="train"):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.filenames = filenames
        self.mode = mode

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_path = os.path.join(self.input_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        recon = np.load(input_path).astype(np.float32)
        gt = np.load(label_path).astype(np.float32)

        if self.mode == "train":
            recon, gt = maybe_flip(recon, gt)


            if np.random.rand() < 0.5:
                recon = random_intensity_scale(recon, 0.9, 1.1)
            if np.random.rand() < 0.5:
                recon = add_gaussian_noise(recon, std=0.015)
            if np.random.rand() < 0.3:
                recon = gaussian_blur_3d(recon, sigma_range=(0.3, 0.7))

            recon = np.clip(recon, 0.0, 1.0)

        input_tensor = torch.from_numpy(recon).float()
        label_tensor = torch.from_numpy(gt).float()
        return input_tensor, label_tensor, filename


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


def hybrid_loss(pred, target):
    ssim = pytorch_msssim.ssim(pred.unsqueeze(1), target.unsqueeze(1), data_range=1.0)
    mse = nn.functional.mse_loss(pred, target)
    return (0.2 * (1 - ssim) + 0.8 * mse), ssim, mse


def evaluate(model, dataloader, criterion, device):
    """Runs a full pass with no grad updates, returns average hybrid loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss, _, _ = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device,
          best_weights_path="unet_best_weights.pt", final_weights_path="unet_final_weights.pt"):
    model.to(device)
    hybrid_batch_losses = []
    ssim_batch_losses = []
    mse_batch_losses = []
    val_losses = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        hybrid_running_loss = 0.0
        ssim_running_loss = 0.0
        mse_running_loss = 0.0

        for i, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            hybrid_loss_val, ssim_loss, mse_loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))
            hybrid_loss_val.backward()
            optimizer.step()

            hybrid_batch_losses.append(hybrid_loss_val.item())
            ssim_batch_losses.append(ssim_loss.item())
            mse_batch_losses.append(mse_loss.item())

            hybrid_running_loss += hybrid_loss_val.item()
            ssim_running_loss += ssim_loss.item()
            mse_running_loss += mse_loss.item()

            if i % 5 == 4:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Hybrid Loss: {hybrid_running_loss / 5:.6f}, SSIM Loss: {ssim_running_loss / 5:.6f}, MSE Loss: {mse_running_loss / 5:.6f}")
                hybrid_running_loss = 0.0
                ssim_running_loss = 0.0
                mse_running_loss = 0.0

        print(f"Epoch {epoch+1}, Hybrid Loss: {hybrid_running_loss}, SSIM Loss: {ssim_running_loss}, MSE Loss: {mse_running_loss}")

        # --- Validation pass: decide if this epoch produced the best model ---
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Validation Hybrid Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> New best model (val loss {val_loss:.6f}), saved to {best_weights_path}")

    # Always save the final-epoch weights too, for comparison
    torch.save(model.state_dict(), final_weights_path)
    print(f"Saved final-epoch weights to {final_weights_path}")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

    plt.figure(figsize=(10, 4))
    plt.plot(hybrid_batch_losses, linewidth=2)
    plt.title('Hybrid Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig("u-net-hybrid-training_loss.png")

    plt.figure(figsize=(10, 4))
    plt.plot(ssim_batch_losses, linewidth=2)
    plt.title('SSIM Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig("u-net-ssim-training_loss.png")

    plt.figure(figsize=(10, 4))
    plt.plot(mse_batch_losses, linewidth=2)
    plt.title('MSE Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig("u-net-mse-training_loss.png")

    plt.figure(figsize=(10, 4))
    plt.plot(val_losses, linewidth=2, marker='o')
    plt.title('Validation Hybrid Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig("u-net-val-loss.png")


def test(model, dataloader, criterion, device):
    model.eval()
    model.to(device)
    hybrid_total_loss = 0.0
    ssim_total_loss = 0.0
    mse_total_loss = 0.0

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            hybrid_loss_val, ssim_loss, mse_loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))

            hybrid_total_loss += hybrid_loss_val.item()
            ssim_total_loss += ssim_loss.item()
            mse_total_loss += mse_loss.item()

    print(f"Average hybrid test loss: {hybrid_total_loss / len(dataloader):.6f}")
    print(f"Average ssim test loss: {ssim_total_loss / len(dataloader):.6f}")
    print(f"Average mse test loss: {mse_total_loss / len(dataloader):.6f}")


def visualize_predictions(model, device, dataloader, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axs = [axs]

    with torch.no_grad():
        count = 0
        for inputs, labels, filenames_batch in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for i in range(inputs.size(0)):
                if count >= num_samples:
                    break
                inp = inputs[i].cpu().numpy()
                out = outputs[i].cpu().numpy()
                lbl = labels[i].cpu().numpy()
                fname = filenames_batch[i]

                axs[count][0].imshow(inp[32, :, :], cmap='jet', vmin=0.0, vmax=1.0)
                axs[count][0].set_title(f"Input (Reconstructed)\n{fname}")
                axs[count][1].imshow(out[32, :, :], cmap='jet', vmin=0.0, vmax=1.0)
                axs[count][1].set_title("Prediction (Denoised)")
                axs[count][2].imshow(lbl[32, :, :], cmap='jet', vmin=0.0, vmax=1.0)
                axs[count][2].set_title("Ground Truth")
                for ax in axs[count]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                count += 1
            if count >= num_samples:
                break
    plt.tight_layout(pad=1.5)
    plt.savefig("unet_limited_predictions.png")
    print(f"\nSaved {num_samples} visualized predictions")


base_dir = "BASE_DIRECTORY"
input_dir = os.path.join(base_dir, "Recon_Centered_Synthesized_64x64x64_KWBP_x")
label_dir = os.path.join(base_dir, "Ground_Truth_Centered_Synthesized_64x64x64")


print("Before Getting Files")

all_filenames = sorted([
    f for f in os.listdir(input_dir)
    if f.endswith(".npy") and os.path.exists(os.path.join(label_dir, f))
])

print("Start Dataset Construction")
print(f"Found {len(all_filenames)} matched input/label pairs")

# --- Three-way split: train / val / test ---
# 1. Carve off the test set first (held out completely, never used to pick weights)
trainval_filenames, test_filenames = train_test_split(
    all_filenames, test_size=0.2, random_state=42
)
# 2. Split the remainder into train / val (val used only to pick "best" checkpoint)
train_filenames, val_filenames = train_test_split(
    trainval_filenames, test_size=0.2, random_state=42
)

print(f"Train: {len(train_filenames)}, Val: {len(val_filenames)}, Test: {len(test_filenames)}")

train_dataset = PairedNPYDataset(input_dir, label_dir, train_filenames, mode="train")
val_dataset = PairedNPYDataset(input_dir, label_dir, val_filenames, mode="val")
test_dataset = PairedNPYDataset(input_dir, label_dir, test_filenames, mode="test")

print("Dataloader")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

input_sample, label_sample, sample_filename = train_dataset[0]
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Input shape: {input_sample.shape}")
print(f"Label shape: {label_sample.shape}")
print(f"Sample filename: {sample_filename}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet()
model.to(device)

criterion = hybrid_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, val_loader, criterion, optimizer, epochs=300, device=device)

model.load_state_dict(torch.load("unet_best_weights.pt"))
test(model, test_loader, criterion, device)
visualize_predictions(model, device, test_loader, num_samples=5)
