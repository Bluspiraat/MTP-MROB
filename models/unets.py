import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class EarlyFusionUNet(nn.Module):
    """U-Net with early fusion for RGB + DSM inputs.
    Concatenates DSM as an additional input channel to RGB before feeding into a pretrained encoder.
    Works with any input size (multiples of 32)."""

    def __init__(self, n_classes=14, encoder_name='resnet34', pretrained=True):
        super().__init__()  # Make sure that it inherits information from the PyTorch Module class. It calls the
        # __init__ of nn.Modules

        weights = 'imagenet' if pretrained else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=4,  # 3 RGB + 1 DSM
            classes=n_classes
        )

    def forward(self, rgb_dsm):  # This method is never called directly, it is handeled by the nn.Module by calling
        # EarlyFusionUNet.__call__() which handles hooks, train and evaluation modes and calls forward internally
        # Early fusion: concatenate DSM to RGB along the channel dimension
        logits = self.model(rgb_dsm)
        return logits


class RGBUNet(nn.Module):
    """U-Net without any fusion.
     Works with any input size (multiples of 32)."""

    def __init__(self, n_classes=14, encoder_name='resnet34', pretrained=True):
        super().__init__()

        weights = 'imagenet' if pretrained else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,  # 3 RGB
            classes=n_classes
        )

    def forward(self, rgb):
        logits = self.model(rgb)
        return logits


class PretrainedMidFusionUNet(nn.Module):
    """Dual-encoder U-Net for RGB + DSM fusion (mid-level fusion).
    Works with variable input sizes (multiples of 32)."""

    def __init__(self, n_classes=14, rgb_encoder='resnet34', dsm_encoder='resnet18'):
        super().__init__()

        # Encoders
        self.encoder_rgb = smp.encoders.get_encoder(
            rgb_encoder, in_channels=3, depth=5, weights='imagenet'
        )
        self.encoder_dsm = smp.encoders.get_encoder(
            dsm_encoder, in_channels=1, depth=5, weights=None
        )

        # Fused decoder (input channels = sum of encoder channel counts)
        fused_channels = [a + b for a, b in zip(
            self.encoder_rgb.out_channels, self.encoder_dsm.out_channels)]

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=fused_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=True,
        )

        # Output head
        self.segmentation_head = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, rgb, dsm):
        feats_rgb = self.encoder_rgb(rgb)
        feats_dsm = self.encoder_dsm(dsm)

        fused_feats = [torch.cat([f_r, f_d], dim=1)
                       for f_r, f_d in zip(feats_rgb, feats_dsm)]

        x = self.decoder(*fused_feats)
        logits = self.segmentation_head(x)
        return logits