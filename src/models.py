"""Teacher-specified FC baseline and a compact convolutional baseline."""

import torch
from torch import nn


class FullyConnectedDAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("Expected input shape (N, 1, 28, 28)")
        return self.decoder(self.encoder(images.flatten(start_dim=1))).reshape_as(images)


class ConvolutionalDAE(nn.Module):
    """28->14->7->14->28 with a linear reconstruction head.

    The auxiliary input is constant zero in the unconditioned control, or a known
    sigma map in the conditioned variant, with exactly the same parameter count.
    Raw outputs are used for training; evaluation explicitly clips to [0, 1].
    """

    def __init__(self, base_channels: int = 16, conditioned: bool = False, residual: bool = False):
        super().__init__()
        if type(base_channels) is not int or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")
        if type(conditioned) is not bool:
            raise ValueError("conditioned must be a boolean")
        if type(residual) is not bool:
            raise ValueError("residual must be a boolean")
        self.base_channels = base_channels
        self.conditioned = conditioned
        self.residual = residual
        width = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

    def forward(self, images, sigma=None):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("Expected input shape (N, 1, 28, 28)")
        if self.conditioned:
            if sigma is None:
                raise ValueError("The conditioned model requires a known sigma for each input")
            values = torch.as_tensor(sigma, device=images.device, dtype=images.dtype)
            if values.ndim == 0:
                values = values.expand(len(images))
            if (values.shape != (len(images),) or not torch.isfinite(values).all()
                    or (values < 0).any()):
                raise ValueError("sigma must be a nonnegative finite scalar or a length-N vector")
            auxiliary = values.reshape(-1, 1, 1, 1).expand_as(images)
        else:
            auxiliary = torch.zeros_like(images)
        inputs = torch.cat([images, auxiliary], dim=1)
        raw = self.decoder(self.encoder(inputs))
        return images - raw if self.residual else raw


def build_model(config):
    if config["type"] == "fc":
        return FullyConnectedDAE(config["latent_dim"])
    if config["type"] == "cae":
        return ConvolutionalDAE(config["base_channels"], residual=config.get("residual", False))
    if config["type"] == "cae_conditioned":
        return ConvolutionalDAE(config["base_channels"], conditioned=True,
                                residual=config.get("residual", False))
    raise ValueError(f"Unsupported model type: {config['type']}")


def predict_clean(model, noisy, sigma):
    if getattr(model, "conditioned", False):
        return model(noisy, sigma=sigma)
    return model(noisy)


def model_description(model):
    return {
        "class": type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": [None, 1, 28, 28],
        "output_shape": [None, 1, 28, 28],
        "training_output": "sigmoid_in_0_1" if isinstance(model, FullyConnectedDAE) else "linear_unclipped",
        "evaluation_output": "clip_to_0_1",
        "auxiliary_channel": (("known_sigma" if model.conditioned else "constant_zero")
                              if isinstance(model, ConvolutionalDAE) else None),
        "requires_known_noise_strength": getattr(model, "conditioned", False),
        "prediction_target": "effective_noise_input_minus_clean" if getattr(model, "residual", False) else "clean_image",
        "note": "FC and CAE differ in architecture and output head; this comparison alone does not isolate convolution as a causal factor.",
    }
