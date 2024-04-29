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
    with torch.no_grad():
        for watermarked_images, original_images in data_loader:
            if images_logged >= num_images:
                break
            watermarked_images = watermarked_images.to(device)
            outputs = model(watermarked_images)
            # Log pairs of images: input and output
            for i in range(min(num_images - images_logged, watermarked_images.size(0))):
                # Assuming outputs and inputs are torch Tensors scaled to [0, 1]
                img_pair = torch.cat(
                    [watermarked_images[i].unsqueeze(0), outputs[i].unsqueeze(0)], dim=0
                )
                logged_images.append(
                    wandb.Image(img_pair, caption=f"Epoch {epoch} Pair {i+1}")
                )
                images_logged += 1

    wandb.log({"reconstructions": logged_images})
