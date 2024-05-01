# %%
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from logger import Logger, log_images

# pixels should be scaled to be between 0 and 1.
# from dataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# model = ConvAutoencoder().to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_discriminator(
    epochs,
    train_loader,
    val_loader,
    generator,  # not trained
    discriminator,  # trained
    loss_fn,
    optimizer,  # just for discriminator
    device,
    experiment_name="discriminator_experiment1",
    checkpoint_path="checkpoints",
):
    config = {"epochs": epochs, "optimizer": type(optimizer).__name__}
    logger = Logger(experiment_name, config=config)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in tqdm(range(epochs), desc="Training the discriminator", leave=True):
        discriminator.train()
        running_loss = 0.0
        running_real_loss = 0.0
        running_fake_loss = 0.0
        for watermarked_images, original_images in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        ):
            watermarked_images = watermarked_images.to(device)
            original_images = original_images.to(device)

            # Real images
            real_labels = torch.ones(original_images.size(0), 1, device=device)
            discriminator.zero_grad()
            real_preds = discriminator(original_images)
            real_loss = loss_fn(real_preds, real_labels)
            running_real_loss += real_loss.item()

            # Generated images
            fake_imgs = generator(watermarked_images).detach()
            fake_labels = torch.zeros(fake_imgs.size(0), 1, device=device)
            fake_preds = discriminator(fake_imgs)
            fake_loss = loss_fn(fake_preds, fake_labels)
            running_fake_loss += fake_loss.item()

            # Combine losses and backpropagate
            total_loss = real_loss + fake_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        avg_loss = running_loss / len(train_loader)
        avg_real_loss = running_real_loss / len(train_loader) / 2
        avg_fake_loss = running_fake_loss / len(train_loader) / 2

        # Validation loss
        val_loss = 0.0
        real_val_loss = 0.0
        fake_val_loss = 0.0
        discriminator.eval()
        with torch.no_grad():
            for watermarked_images, original_images in tqdm(
                val_loader, desc=f"Validation: Epoch {epoch + 1}", leave=False
            ):
                watermarked_images = watermarked_images.to(device)
                original_images = original_images.to(device)

                real_labels = torch.ones(original_images.size(0), 1, device=device)
                real_preds = discriminator(original_images)
                real_loss = loss_fn(real_preds, real_labels)
                real_val_loss += real_loss.item()

                fake_imgs = generator(watermarked_images).detach()
                fake_labels = torch.zeros(fake_imgs.size(0), 1, device=device)
                fake_preds = discriminator(fake_imgs)
                fake_loss = loss_fn(fake_preds, fake_labels)
                fake_val_loss += fake_loss.item()

                total_val_loss = real_loss + fake_loss
                val_loss += total_val_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_real_loss = real_val_loss / len(val_loader) / 2
        avg_val_fake_loss = fake_val_loss / len(val_loader) / 2
        logger.log(
            {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "real_train_loss": avg_real_loss,
                "real_val_loss": avg_val_real_loss,
                "fake_train_loss": avg_fake_loss,
                "fake_val_loss": avg_val_fake_loss,
                "epoch": epoch + 1,
            }
        )

        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth"
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
        torch.save(
            {
                "epoch": epoch + 1,
                "discriminator_state_dict": discriminator.state_dict(),
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

from logger import Logger, log_images

# Define the indices for the subsets
train_indices = range(200)  # First 200 examples for training
val_indices = range(20)  # First 20 examples for validation

# Create subset datasets
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

# Create data loaders for real and synthetic images
train_loader = DataLoader(dataset=train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=16, shuffle=False)
# load the generator and instantiate the discriminator
# %%
from models import ConvAutoencoder, Discriminator

generator = ConvAutoencoder().to(device)
discriminator = Discriminator().to(device)
optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
generator.load_state_dict(
    torch.load("checkpoints_CAE/checkpoint_epoch_39.pth")["model_state_dict"]
)

# %%

# change the inputs are a bit dif
train_discriminator(
    epochs=10,
    train_loader=train_loader,
    val_loader=val_loader,
    generator=generator,
    discriminator=discriminator,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
)
# %%
