# %%
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.up1 = DoubleConv(768, 256)  # Adjusted channel sizes for concatenation
        self.up2 = DoubleConv(384, 128)
        self.up3 = DoubleConv(192, 64)
        self.outc = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = F.max_pool2d(x1, 2)
        x3 = self.down1(x2)
        x4 = F.max_pool2d(x3, 2)
        x5 = self.down2(x4)
        x6 = F.max_pool2d(x5, 2)
        x7 = self.down3(x6)

        # Ensure dimensions match for skip connections
        x = F.interpolate(x7, size=x5.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, x5], dim=1)
        x = self.up1(x)

        x = F.interpolate(x, size=x3.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_mu = nn.Conv2d(
            256, 128, kernel_size=1
        )  # Output mu directly from feature maps
        self.conv_logvar = nn.Conv2d(256, 128, kernel_size=1)  # Output logvar directly

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        mu = self.conv_mu(x)
        log_var = self.conv_logvar(x)
        z = self.reparameterize(mu, log_var)
        z = F.relu(self.deconv1(z), inplace=True)
        z = F.relu(self.deconv2(z), inplace=True)
        z = F.relu(self.deconv3(z), inplace=True)
        z = F.interpolate(
            z, size=(500, 500), mode="bilinear", align_corners=False
        )  # Correct size adjustment
        z = torch.sigmoid(z)  # Normalize to [0,1]
        return z, mu, log_var


# %%
