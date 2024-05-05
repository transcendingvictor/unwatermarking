# This script is to train a GAN with generator not using reconstruction loss, just trying to fool discriminator.
import os

import piq  # For SSIM (metric)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from dataset import CustomDataset
from logger import Logger, log_images
from models import ConvAutoencoder, Discriminator


def train_GAN(
    epochs,
    train_loader,
    val_loader,
    generator,  #  trained
    discriminator,  # trained
    loss_fn,
    optimizerD,  # discriminator
    optimizerG,  # generator
    device,
    reconstruction_weight=0,
    experiment_name="GAN_experiment3",
    checkpoint_path="checkpoints",
    discriminator_update_ratio=3,
):
    config = {
        "epochs": epochs,
        "optimizerD": type(optimizerD).__name__,
        "optimizerG": type(optimizerG).__name__,
    }
    logger = Logger(experiment_name, config=config)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in tqdm(range(epochs), desc="Training the GAN", leave=True):
        discriminator.train()
        running_Dloss = 0.0
        running_real_loss = 0.0
        running_fake_loss = 0.0
        running_fool_loss = 0.0
        running_reconstruction_loss = 0.0
        running_Gloss = 0.0

        generator.train()
        discriminator.train()

        discriminator_counter = 0
        for watermarked_images, original_images in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        ):
            watermarked_images = watermarked_images.to(device)
            original_images = original_images.to(device)

            # Discriminator loss on real images. Label: 1
            real_labels = torch.ones(original_images.size(0), 1, device=device)
            real_preds = discriminator(original_images)
            real_loss = loss_fn(real_preds, real_labels)
            running_real_loss += real_loss.item()

            # Discriminator loss on generated images. Label: 0
            fake_imgs = generator(
                watermarked_images
            ).detach()  # not mess with gradients
            fake_labels = torch.zeros(fake_imgs.size(0), 1, device=device)
            fake_preds = discriminator(fake_imgs)
            fake_loss = loss_fn(fake_preds, fake_labels)
            running_fake_loss += fake_loss.item()

            # Train discriminator
            total_loss = real_loss + fake_loss
            adjusted_loss = total_loss / discriminator_update_ratio
            running_Dloss += total_loss.item()

            if discriminator_counter % discriminator_update_ratio == 0:
                optimizerD.zero_grad()
                adjusted_loss.backward()
                optimizerD.step()

            discriminator_counter += 1

            # Generator loss fooling discriminator. Label: 1
            optimizerG.zero_grad()
            fake_imgs = generator(watermarked_images)
            fake_preds = discriminator(fake_imgs)
            fool_loss = loss_fn(fake_preds, real_labels)
            running_fool_loss += fool_loss.item()

            # Generator reconstruction loss.
            reconstruction_loss = torch.nn.L1Loss()(fake_imgs, original_images)
            running_reconstruction_loss += reconstruction_loss.item()

            # Train generator.
            Gloss = fool_loss + reconstruction_weight * reconstruction_loss
            Gloss.backward()
            optimizerG.step()
            running_Gloss += Gloss.item()

        avg_Dloss = running_Dloss / len(train_loader)
        avg_real_loss = running_real_loss / len(train_loader) / 2
        avg_fake_loss = running_fake_loss / len(train_loader) / 2
        avg_fool_loss = running_fool_loss / len(train_loader)
        avg_reconstruction_loss = running_reconstruction_loss / len(train_loader)
        avg_Gloss = running_Gloss / len(train_loader)

        # Validation loss
        val_Dloss = 0.0
        real_val_loss = 0.0
        fake_val_loss = 0.0
        fool_val_loss = 0.0
        reconstruction_val_loss = 0.0
        val_Gloss = 0.0
        val_ssim = 0.0  # To accumulate SSIM scores

        discriminator.eval()
        generator.eval()
        with torch.no_grad():
            for watermarked_images, original_images in tqdm(
                val_loader, desc=f"Validation: Epoch {epoch + 1}", leave=False
            ):
                watermarked_images = watermarked_images.to(device)
                original_images = original_images.to(device)

                # Evaluate the discriminator's ability to classify real as real
                real_labels = torch.ones(original_images.size(0), 1, device=device)
                real_preds = discriminator(original_images)
                real_loss = loss_fn(real_preds, real_labels)
                real_val_loss += real_loss.item()

                # Evaluate the discriminator's ability to classify fake as fake
                fake_imgs = generator(watermarked_images)
                fake_labels = torch.zeros(fake_imgs.size(0), 1, device=device)
                fake_preds = discriminator(fake_imgs)
                fake_loss = loss_fn(fake_preds, fake_labels)
                fake_val_loss += fake_loss.item()

                totalDloss = real_loss + fake_loss
                val_Dloss += totalDloss.item()

                # Measure how well the generator is fooling the discriminator
                fool_loss = loss_fn(fake_preds, real_labels)
                fool_val_loss += fool_loss.item()

                # Compute the reconstruction loss
                reconstruction_loss = torch.nn.L1Loss()(fake_imgs, original_images)
                reconstruction_val_loss += reconstruction_loss.item()

                # Combine losses for generator's total validation loss
                Gloss = fool_loss + reconstruction_weight * reconstruction_loss
                val_Gloss += Gloss.item()

                # Calculate and accumulate SSIM
                current_ssim = piq.ssim(fake_imgs, original_images, data_range=1.0)
                val_ssim += current_ssim.item()

        avg_val_Dloss = val_Dloss / len(val_loader)
        avg_val_real_loss = real_val_loss / len(val_loader)
        avg_val_fake_loss = fake_val_loss / len(val_loader)
        avg_val_fool_loss = fool_val_loss / len(val_loader)
        avg_val_reconstruction_loss = reconstruction_val_loss / len(val_loader)
        avg_val_Gloss = val_Gloss / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)  # Average SSIM over the dataset

        # Log all metrics
        logger.log(
            {
                "train_Dloss": avg_Dloss,
                "train_Gloss": avg_Gloss,
                "val_Dloss": avg_val_Dloss,
                "val_Gloss": avg_val_Gloss,
                "real_train_loss": avg_real_loss,
                "real_val_loss": avg_val_real_loss,
                "fake_train_loss": avg_fake_loss,
                "fake_val_loss": avg_val_fake_loss,
                "fool_train_loss": avg_fool_loss,
                "fool_val_loss": avg_val_fool_loss,
                "reconstruction_train_loss": avg_reconstruction_loss,
                "reconstruction_val_loss": avg_val_reconstruction_loss,
                "SSIM_val": avg_val_ssim,  # Include SSIM in your logging
                "epoch": epoch + 1,
            }
        )

        # log some images
        log_images(epoch, generator, device, val_loader, num_images=5)

        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_GAN3_epoch{epoch+1}.pth"
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
        torch.save(
            {
                "epoch": epoch + 1,
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
                "train_Dloss": avg_Dloss,
                "val_Dloss": avg_val_Dloss,
                "generator_state_dict": generator.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "train_Gloss": avg_Gloss,
                "val_Gloss": avg_val_Gloss,
            },
            checkpoint_filepath,
        )

    logger.finish()


# Load the watermark (6x1min) and the original (6x1min) datasets from HF.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loss_fn = torch.nn.BCEWithLogitsLoss()

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

    generator = ConvAutoencoder().to(device)
    discriminator = Discriminator().to(device)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-4)

    generator.load_state_dict(
        torch.load("checkpoints/checkpoints_CAE/checkpoint_epoch_39.pth")[
            "model_state_dict"
        ]
    )
    optimizerG.load_state_dict(
        torch.load("checkpoints/checkpoints_CAE/checkpoint_epoch_39.pth")[
            "optimizer_state_dict"
        ]
    )

    train_GAN(
        epochs=60,
        train_loader=train_loader,
        val_loader=val_loader,
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        optimizerD=optimizerD,
        optimizerG=optimizerG,
        device=device,
    )
