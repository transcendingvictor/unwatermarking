import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from dataset import CustomDataset
from logger import Logger, log_images_vae
from models import VAE


# Loss components
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train_vae(
    epochs,
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    device,
    kl_weight=0,
    experiment_name="vae_kl_experiment",
    checkpoint_path="checkpoints_vae_kl",
):
    config = {"epochs": epochs, "optimizer": type(optimizer).__name__}
    logger = Logger(experiment_name, config=config)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in tqdm(range(21, 21 + epochs), desc="Training the VAE", leave=True):
        model.train()
        running_loss = 0.0
        ruuning_rec_loss = 0.0
        running_kl_loss = 0.0
        for watermarked_images, original_images in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        ):
            watermarked_images = watermarked_images.to(device)
            original_images = original_images.to(device)

            # Forward pass
            reconstructed_images, mu, log_var = model(watermarked_images)
            rec_loss = loss_fn(reconstructed_images, original_images)
            kl_loss = kl_weight * kl_divergence(mu, log_var)
            loss = rec_loss + kl_loss  # Total loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ruuning_rec_loss += rec_loss.item()
            running_kl_loss += kl_loss.item()

        avg_loss = running_loss / len(train_loader)
        avg_rec_loss = ruuning_rec_loss / len(train_loader)
        avg_kl_loss = running_kl_loss / len(train_loader)
        # Validation loss
        val_loss = 0.0
        val_rec_loss = 0.0
        val_kl_loss = 0.0
        model.eval()
        with torch.no_grad():
            for watermarked_images, original_images in tqdm(
                val_loader, desc=f"Validation: Epoch {epoch + 1}", leave=False
            ):
                watermarked_images = watermarked_images.to(device)
                original_images = original_images.to(device)
                reconstructed_images, mu, log_var = model(watermarked_images)
                rec_loss = loss_fn(reconstructed_images, original_images)
                kl_loss = kl_divergence(mu, log_var)
                val_loss += (rec_loss + kl_loss).item()
                val_kl_loss += kl_loss.item()
                val_rec_loss += rec_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_rec_loss = val_rec_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)
        logger.log(
            {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "train_rec_loss": avg_rec_loss,
                "val_rec_loss": avg_val_rec_loss,
                "train_kl_loss": avg_kl_loss,
                "val_kl_loss": avg_val_kl_loss,
                "epoch": epoch + 1,
            }
        )
        log_images_vae(epoch, model, device, val_loader, num_images=5)

        # Save checkpoint
        checkpoint_filename = f"checkpoint_VAE_epoch_{epoch+1}.pth"
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
    model = VAE().to(device)
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

    # Load the checkpoints of the model to train it further
    # checkpoint = torch.load("checkpoints/checkpoints_vae/checkpoint_VAE_epoch_39.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_vae(
        epochs=10,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
