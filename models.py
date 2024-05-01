# Where the VAE, the GAN, and the Diffussion Model are defined.

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"The new Models 802 is imported")


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # Bottleneck.

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # Use sigmoid if image pixels are normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(
            x, size=(500, 500)
        )  # Ensure output matches input dimensions exactly
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            # three image channels
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        )

        self.num_flat_features = self._get_conv_output()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(self.num_flat_features, 1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

    def _get_conv_output(self):
        input = torch.rand(1, 3, 500, 500)  # size of the images
        output = self.conv_layers(input)
        return int(torch.prod(torch.tensor(output.shape[1:])).item())


