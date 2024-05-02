import torch

import wandb


class Logger:
    def __init__(self, experiment_name, project="autoencoder_project", config=None):
        self.run = wandb.init(project=project, name=experiment_name, config=config)

    def log(self, metrics, step=None):
        self.run.log(metrics, step=step)

    def finish(self):
        self.run.finish()


def log_images(epoch, model, device, data_loader, num_images=10):
    model.eval()
    images_logged = 0
    logged_images = []
    # desired_indices = [3, 93, 182, 272, 363]  # indices of images you want to log
    desired_indices = [1, 2, 4, 8, 14]
    current_index = 0  # to track the index of the images being processed

    with torch.no_grad():
        for watermarked_images, original_images in data_loader:
            watermarked_images = watermarked_images.to(device)
            outputs = model(watermarked_images)
            # Log pairs of images: input and output
            for i in range(watermarked_images.size(0)):
                if current_index in desired_indices:
                    img_pair = torch.cat(
                        [watermarked_images[i].unsqueeze(0), outputs[i].unsqueeze(0)],
                        dim=0,
                    )
                    logged_images.append(
                        wandb.Image(
                            img_pair, caption=f"Epoch {epoch} Pair {current_index + 1}"
                        )
                    )
                    images_logged += 1
                    if images_logged >= num_images:
                        break
                current_index += 1
            if images_logged >= num_images:
                break

    wandb.log({"reconstructions": logged_images})


def log_images_diffusion(
    epoch, model, device, data_loader, num_images=10, num_steps=10
):
    model.eval()
    images_logged = 0
    logged_images = []
    # desired_indices = [3, 93, 182, 272, 363]  # indices of images you want to log
    desired_indices = [1, 2, 4, 8, 14]  # indices of images you want to log
    current_index = 0  # to track the index of the images being processed

    with torch.no_grad():
        for watermarked_images, original_images in data_loader:
            watermarked_images = watermarked_images.to(device)
            outputs = model(watermarked_images)
            for _ in range(num_steps - 1):  # only change wrt the above
                outputs = model(outputs)
            # Log pairs of images: input and output
            for i in range(watermarked_images.size(0)):
                if current_index in desired_indices:
                    img_pair = torch.cat(
                        [watermarked_images[i].unsqueeze(0), outputs[i].unsqueeze(0)],
                        dim=0,
                    )
                    logged_images.append(
                        wandb.Image(
                            img_pair, caption=f"Epoch {epoch} Pair {current_index + 1}"
                        )
                    )
                    images_logged += 1
                    if images_logged >= num_images:
                        break
                current_index += 1
            if images_logged >= num_images:
                break

    wandb.log({"reconstructions": logged_images})
