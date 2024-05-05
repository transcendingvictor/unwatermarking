import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from dataset import CustomDataset
from logger import Logger, log_images
from models import ConvAutoencoder

# sys.path.append(os.path.abspath("../"))


def train_autoencoder(
    epochs,
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    device,
    experiment_name="autoencoder_experiment",
    checkpoint_path="checkpoints",
):
    config = {"epochs": epochs, "optimizer": type(optimizer).__name__}
    logger = Logger(experiment_name, config=config)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in tqdm(range(epochs), desc="Training the autoencoder", leave=True):
        model.train()
        running_loss = 0.0
        for watermarked_images, original_images in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        ):
            watermarked_images = watermarked_images.to(device)
            original_images = original_images.to(device)

            # Forward pass
            outputs = model(watermarked_images)
            loss = loss_fn(outputs, original_images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        # Validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for watermarked_images, original_images in tqdm(
                val_loader, desc=f"Validation: Epoch {epoch + 1}", leave=False
            ):
                watermarked_images = watermarked_images.to(device)
                original_images = original_images.to(device)
                outputs = model(watermarked_images)
                loss = loss_fn(outputs, original_images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.log(
            {"train_loss": avg_loss, "val_loss": avg_val_loss, "epoch": epoch + 1}
        )
        log_images(epoch, model, device, val_loader, num_images=5)

        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth"
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_loss": avg_val_loss,
            },
            checkpoint_filepath,
        )
    logger.finish()


# Load the watermark (6x1min) and the original (6x1min) datasets from HF.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ConvAutoencoder().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    transforms = ToTensor()
    train_dataset = CustomDataset(
        "transcendingvictor/watermark1_flowers_dataset",
        "transcendingvictor/original_flowers_dataset",
        "train",
        transforms,
    )

    val_dataset = CustomDataset(
        "transcendingvictor/watermark1_flowers_dataset",
        "transcendingvictor/original_flowers_dataset",
        "test",
        transforms,
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

    train_autoencoder(
        epochs=40,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
