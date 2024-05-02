# %% This script is to train a GAN with generator using reconstruction loss too.
import os
import random

import piq  # For SSIM (metric)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from logger import Logger, log_images, log_images_diffusion
from models import UNet

# pixels should be scaled to be between 0 and 1.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet(n_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


def interpolate_images(watermarked, original, t):
    return (1 - t) * original + t * watermarked


def train_diffusion_model(
    epochs,
    train_loader,
    model,
    loss_fn,
    optimizer,
    device,
    num_steps=10,  # Number of diffusion steps
    experiment_name="diffusion_experiment",
    checkpoint_path="checkpoints_diffusion",
):
    config = {
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
    }
    logger = Logger(experiment_name, config=config)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Generate t values ranging from 1.0 to a small value above 0 (not including fully clean)
    input_t_values = torch.linspace(1.0, 0.1, num_steps)
    target_t_values = torch.linspace(
        0.9, 0.0, num_steps
    )  # Target t values include the clean state

    for epoch in tqdm(range(epochs), desc="Training the diffusion model"):
        model.train()
        running_loss = 0.0

        for watermarked_images, original_images in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        ):
            watermarked_images = watermarked_images.to(device)
            original_images = original_images.to(device)

            # Randomly select an index for diffusion steps
            idx = random.randint(0, num_steps - 1)

            # Interpolate images based on the selected t value
            inputs = interpolate_images(
                watermarked_images, original_images, input_t_values[idx]
            ).to(device)
            targets = interpolate_images(
                watermarked_images, original_images, target_t_values[idx]
            ).to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation loss
        val_loss = 0.0
        val_ssim = 0.0  # To accumulate SSIM scores

        with torch.no_grad():
            for watermarked_images, original_images in tqdm(
                val_loader, desc=f"Validation: Epoch {epoch + 1}", leave=False
            ):
                watermarked_images = watermarked_images.to(device)
                original_images = original_images.to(device)

                # Randomly select an index for diffusion steps
                idx = random.randint(0, num_steps - 1)

                # Interpolate images based on the selected t value
                inputs = interpolate_images(
                    watermarked_images, original_images, input_t_values[idx]
                ).to(device)
                targets = interpolate_images(
                    watermarked_images, original_images, target_t_values[idx]
                ).to(device)

                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item()
                val_loss += running_loss

                # Calculate and accumulate SSIM
                current_ssim = piq.ssim(outputs, inputs, data_range=1.0)
                val_ssim += current_ssim.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)  # Average SSIM over the dataset

        # Log all metrics
        logger.log(
            {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "SSIM_val": avg_val_ssim,  # Include SSIM in your logging
                "epoch": epoch + 1,
            }
        )
        # log some images
        # log_images(epoch, model, device, val_loader, num_images=5)
        log_images_diffusion(
            epoch, model, device, val_loader, num_images=5, num_steps=num_steps
        )
        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_diffussion_epoch{epoch+1}.pth"
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
            },
            checkpoint_filepath,
        )

    logger.finish()


# %%
# Load the watermark (6x1min) and the original (6x1min) datasets from HF.
# if __name__ == "__main__":
from dataset import CustomDataset

transforms = ToTensor()
train_dataset = CustomDataset(
    "transcendingvictor/watermark1_flowers_dataset",
    "transcendingvictor/original_flowers_dataset",
    "train",
    transforms,
)
# %%
val_dataset = CustomDataset(
    "transcendingvictor/watermark1_flowers_dataset",
    "transcendingvictor/original_flowers_dataset",
    "test",
    transforms,
)
# %%

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
# %% to reload modules
import importlib

import models

# Make sure this is imported if not already done

importlib.reload(models)
# %% for a subset of data
from torch.utils.data import DataLoader, Subset

# Define the indices for the subsets
train_indices = range(64)  # First 200 examples for training
val_indices = range(16)  # First 20 examples for validation

# Create subset datasets
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

# Create data loaders for real and synthetic images
train_loader = DataLoader(dataset=train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=4, shuffle=False)
# load the generator and instantiate the discriminator
# %%
from models import ConvAutoencoder, Discriminator

checkpoint = torch.load("special_checkpoints/checkpoint_GAN2_epoch10.pth")

# model = ConvAutoencoder().to(device)
# generator.load_state_dict(checkpoint["generator_state_dict"])
# optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
# optimizer.load_state_dict(checkpoint["optimizerG_state_dict"])

# %%

train_diffusion_model(
    epochs=5,
    train_loader=train_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
)
# %%
