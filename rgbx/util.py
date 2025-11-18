import os
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import torchvision


def vizualize_intrinsic_channels(
    intrinsic_channels: dict[str, PIL.Image], save_path: str | None = None
) -> None:
    plt.figure(figsize=(20 * 6, 20))

    for i, (aov, image) in enumerate(intrinsic_channels.items()):
        plt.subplot(1, 6, i + 1)
        plt.imshow(image)
        plt.title(aov)
        plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path)


def save_intrinsic_channels(
    intrinsic_channels: dict[str, PIL.Image], name: str, save_dir: str
) -> None:
    if not Path(save_dir).exists():
        os.makedirs(save_dir)

    for aov, image in intrinsic_channels.items():
        save_path: Path = Path(save_dir) / (name + "." + aov + ".png")
        image.save(save_path)


def apply_mask(
    intrinsic_channels: dict[str, PIL.Image], mask_path: str
) -> dict[str, PIL.Image]:
    masked_intrinsic_channels = {}
    mask = torchvision.io.read_image(mask_path)
    mask = mask.float() / 255.0

    for aov, image in intrinsic_channels.items():
        image = torchvision.transforms.functional.pil_to_tensor(image)
        masked_image = (image.float() / 255.0) * mask
        masked_intrinsic_channels[aov] = torchvision.transforms.functional.to_pil_image(
            masked_image, mode=None
        )

    return masked_intrinsic_channels
