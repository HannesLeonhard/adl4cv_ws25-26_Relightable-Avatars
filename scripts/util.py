import PIL
import torch
import torchvision


def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor from sRGB to linear RGB space.

    Parameters:
        img (torch.Tensor): Input tensor in sRGB space, normalized to [0, 1].

    Returns:
        torch.Tensor: Tensor with linear RGB values, clamped to [0, 1].
    """

    linear = torch.empty_like(img)
    linear_mask = img <= 0.04045
    linear[linear_mask] = img[linear_mask] / 12.92

    power_mask = ~linear_mask
    linear[power_mask] = torch.pow((img[power_mask] + 0.055) / 1.055, 2.4)

    return torch.clamp(linear, 0.0, 1.0)


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor from linear RGB to sRGB space.

    Parameters:
        img (torch.Tensor): Input tensor in linear RGB space, normalized to [0, 1].

    Returns:
        torch.Tensor: Tensor with sRGB values, clamped to [0, 1].
    """

    srgb = torch.empty_like(img)
    linear_mask = img <= 0.0031308
    srgb[linear_mask] = img[linear_mask] * 12.92

    power_mask = ~linear_mask
    srgb[power_mask] = (1.055 * torch.pow(img[power_mask], 1.0 / 2.4)) - 0.055

    return torch.clamp(srgb, 0.0, 1.0)


def load_img(path: str, from_srgb: bool = True) -> torch.Tensor:
    """
    Load an image, normalize to [0, 1], and optionally convert from sRGB to linear RGB.

    Parameters:
        path (str): File path to the image.
        from_srgb (bool, optional): If True, converts from sRGB to linear RGB. Defaults to True.

    Returns:
        torch.Tensor: Loaded image (C, H, W) as float32 in [0, 1] range.
    """

    image = torchvision.io.read_image(path)
    image = image.to(torch.float32)
    image = image / 255.0
    if from_srgb:
        image = srgb_to_linear(image)
    return image


def pil_to_normalized_tensor(image: PIL.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a normalized tensor.

    Parameters:
        image (PIL.Image): Input image to convert.

    Returns:
        torch.Tensor: Tensor with values normalized to the range [0, 1].
    """

    tensor: torch.Tensor = torchvision.transforms.functional.pil_to_tensor(image)
    tensor = tensor.to(torch.float32) / 255.0
    return tensor


def save_img(img: torch.Tensor, path: str, from_linear: bool = True):
    """
    Save an image tensor, converting from linear RGB to sRGB if specified.

    Parameters:
        img (torch.Tensor): Image tensor (C, H, W) in [0, 1] range.
        path (str): File path for saving (e.g., 'output.png').
        from_linear (bool, optional): If True, converts from linear to sRGB before saving.
    """

    if from_linear:
        if img.shape[0] == 4:
            # RGBa
            rgb = linear_to_srgb(img[:3])
            alpha = img[3:4]
            img = torch.cat([rgb, alpha], dim=0)
        else:
            # RGB
            img = linear_to_srgb(img)

    torchvision.utils.save_image(img, path)
