import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pytorch_msssim


# =============================
# Dataset
# =============================
class PairedNPYDataset(Dataset):
    def __init__(self, input_dir, label_dir, filenames, transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_path = os.path.join(self.input_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        input_array = np.load(input_path)
        label_array = np.load(label_path)
        input_tensor = torch.from_numpy(input_array).float()
        label_tensor = torch.from_numpy(label_array).float()
        return input_tensor, label_tensor, filename

# =============================
# U-Net Model
# =============================
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu2(out)
        return out

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

# =============================
# MSSIM + MSE Hybrid Loss
# =============================

def hybrid_loss(pred, target):
    ssim = pytorch_msssim.ssim(pred.unsqueeze(1), target.unsqueeze(1), data_range=1.0)
    mse = nn.functional.mse_loss(pred, target)
    return (0.2 * (1 - ssim) + 0.8 * mse), ssim, mse


# =============================
# Validation helper
# =============================
def validate(model, dataloader, criterion, device):
    """Run a no-grad pass over dataloader and return average hybrid/ssim/mse losses."""
    model.eval()
    hybrid_total, ssim_total, mse_total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            hybrid_loss_val, ssim_loss_val, mse_loss_val = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))
            hybrid_total += hybrid_loss_val.item()
            ssim_total += ssim_loss_val.item()
            mse_total += mse_loss_val.item()
    n = len(dataloader)
    model.train()
    return hybrid_total / n, ssim_total / n, mse_total / n


# =============================
# Training / Testing Functions
# =============================
def train(model, dataloader, criterion, optimizer, epochs, device,
          val_loader=None, checkpoint_path="best_model.pth"):
    """
    Trains `model`. If `val_loader` is provided, evaluates on it after every
    epoch and saves the weights with the lowest validation hybrid loss to
    `checkpoint_path`. At the end of training, the best weights are reloaded
    into `model` so callers (test/visualization) use the best checkpoint
    rather than whatever the final epoch happened to produce.
    """
    model.train()
    model.to(device)
    hybrid_batch_losses = []
    ssim_batch_losses = []
    mse_batch_losses = []
    val_hybrid_losses = []

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        hybrid_running_loss = 0.0
        ssim_running_loss = 0.0
        mse_running_loss = 0.0

        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            hybrid_loss_val, ssim_loss_val, mse_loss_val = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))
            hybrid_loss_val.backward()

            optimizer.step()
            hybrid_batch_losses.append(hybrid_loss_val.item())
            ssim_batch_losses.append(ssim_loss_val.item())
            mse_batch_losses.append(mse_loss_val.item())

            hybrid_running_loss += hybrid_loss_val.item()
            ssim_running_loss += ssim_loss_val.item()
            mse_running_loss += mse_loss_val.item()

            if i % 5 == 4:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Hybrid Loss: {hybrid_running_loss / 5:.6f}, "
                      f"SSIM Loss: {ssim_running_loss / 5:.6f}, MSE Loss: {mse_running_loss / 5:.6f}")

                hybrid_running_loss = 0.0
                ssim_running_loss = 0.0
                mse_running_loss = 0.0

        print(f"Epoch {epoch+1}, Hybrid Loss: {hybrid_running_loss}, SSIM Loss: {ssim_running_loss}, MSE Loss: {mse_running_loss}")

        # ---- Validation + checkpointing on best weights ----
        if val_loader is not None:
            val_hybrid, val_ssim, val_mse = validate(model, val_loader, criterion, device)
            val_hybrid_losses.append(val_hybrid)
            print(f"Epoch {epoch+1} Validation -> Hybrid: {val_hybrid:.6f}, SSIM: {val_ssim:.6f}, MSE: {val_mse:.6f}")

            if val_hybrid < best_val_loss:
                best_val_loss = val_hybrid
                best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  New best model (val hybrid loss={val_hybrid:.6f}) saved to {checkpoint_path}")

    if val_loader is not None and best_epoch != -1:
        print(f"\nBest epoch: {best_epoch} with validation hybrid loss {best_val_loss:.6f}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best weights from {checkpoint_path} back into model.")

        plt.figure(figsize=(10, 4))
        plt.plot(val_hybrid_losses, linewidth=2, marker='o')
        plt.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best epoch ({best_epoch})')
        plt.title('Validation Hybrid Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which='both', linestyle='--')
        plt.tight_layout()
        plt.savefig("u-net-val-hybrid-loss.png")

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
            hybrid_loss_val, ssim_loss_val, mse_loss_val = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))

            hybrid_total_loss += hybrid_loss_val.item()
            ssim_total_loss += ssim_loss_val.item()
            mse_total_loss += mse_loss_val.item()

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

# =============================
# Pipeline Execution
# =============================
base_dir = "/path_to/Image_Reconstruction/"
input_dir = os.path.join(base_dir, "Rec_Imgs/all_npy")
label_dir = os.path.join(base_dir, "Ground_Truth_Centered_Synthesized_64x64x64")

print("Before Getting Files")

all_filenames = sorted([
    f for f in os.listdir(input_dir)
    if f.endswith(".npy") and os.path.exists(os.path.join(label_dir, f))
])

print("Start Dataset Construction")

# First split off the held-out test set, then carve a validation set
# out of the remaining training data (used only for checkpoint selection,
# never for gradient updates).
train_val_filenames, test_filenames = train_test_split(all_filenames, test_size=0.2, random_state=42)
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=0.1, random_state=42)

train_dataset = PairedNPYDataset(input_dir, label_dir, train_filenames)
val_dataset   = PairedNPYDataset(input_dir, label_dir, val_filenames)
test_dataset  = PairedNPYDataset(input_dir, label_dir, test_filenames)

print("Dataloader")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

input_sample, label_sample, sample_filename = train_dataset[0]
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Input shape: {input_sample.shape}")
print(f"Label shape: {label_sample.shape}")
print(f"Sample filename: {sample_filename}")

# Train U-Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet()
model.to(device)

criterion = hybrid_loss
# criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, criterion, optimizer, epochs=200, device=device,
      val_loader=val_loader, checkpoint_path="residual_block_best_weights.pt")
test(model, test_loader, criterion, device)
visualize_predictions(model, device, test_loader, num_samples=5)
