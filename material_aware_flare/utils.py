from PIL import Image
import torch
from typing import Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def convert_rgba_to_rgb_pil(image, background_color=(255, 255, 255)):
    """
    Converts an RGBA image to RGB with the specified background color.
    If the image is already in RGB mode, it is returned as is.

    Parameters:
        image (PIL.Image.Image): Input image (RGBA or RGB).
        background_color (tuple): Background color as an RGB tuple. Default is white (255, 255, 255).

    Returns:
        PIL.Image.Image: RGB image.
    """
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    return image


def save_extracted_channels(
    channels: Dict[str, torch.Tensor], batch_indices: torch.Tensor, save_path: Path
):
    """Saves extracted channels (albedo, roughness, irradiance) to disk."""
    for batch_idx in range(channels["albedo"].shape[0]):
        try:
            name_idx = batch_indices[batch_idx].item()
        except:
            name_idx = batch_idx
        for channel_name, tensor in channels.items():
            img_tensor = tensor[batch_idx, ...].cpu().numpy()
            channel_dir = save_path / channel_name
            channel_dir.mkdir(parents=True, exist_ok=True)
            if img_tensor.shape[-1] == 1:
                # Roughness, Specular Intensity (1 channel)
                img_to_save = np.clip(img_tensor[..., 0], 0.0, 1.0)
                plt.imsave(
                    channel_dir / f"{name_idx:04d}.png", img_to_save, cmap="gray"
                )
            else:
                # Albedo, Normal, Irradiance (3 channels)
                img_to_save = np.clip(img_tensor, 0.0, 1.0)
                plt.imsave(channel_dir / f"{name_idx:04d}.png", img_to_save)

