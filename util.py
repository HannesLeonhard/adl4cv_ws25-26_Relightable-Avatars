import matplotlib as plt
import PIL

def vizualize_intrinsic_channels(intrinsic_channels: dict[str, PIL.Image], save_path: str | None = None) -> None:
    plt.figure(figsize=(20 * 6, 20))

    for i, (aov, image) in enumerate(intrinsic_channels.items()):
        plt.subplot(1, 6, i + 1)
        plt.imshow(image)
        plt.title(aov)
        plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path)

