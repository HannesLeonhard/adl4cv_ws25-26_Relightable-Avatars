from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image
import torch
from typing import Dict, Any

from .utils import convert_rgba_to_rgb_pil


def calculate_metrics(
    gen_pil,
    gt_pil,
    lpips_model,
    lpips_transform,
    device,
    mask_np,
    model_dtype=None
):
    """
    Calculates PSNR, SSIM, and LPIPS inside masked image regions.
    Mask is (H, W, 3) float in [0,1], soft mask supported.
    """
    psnr_val, ssim_val, lpips_val = None, None, None

    try:
        gen_pil_rgb = convert_rgba_to_rgb_pil(gen_pil, background_color=(0, 0, 0))
        gt_pil_rgb = convert_rgba_to_rgb_pil(gt_pil, background_color=(0, 0, 0))
        gen_np = np.array(gen_pil_rgb).astype(np.float32)
        gt_np = np.array(gt_pil_rgb).astype(np.float32)
        if gen_np.shape != gt_np.shape:
            gt_pil_rgb = gt_pil_rgb.resize(gen_pil_rgb.size, Image.LANCZOS)
            gt_np = np.array(gt_pil_rgb).astype(np.float32)
        # mask_np: (H, W, 3) float, convert to single channel
        if mask_np.ndim == 3 and mask_np.shape[-1] == 3:
            mask = mask_np.mean(axis=-1)  # (H,W)
        else:
            mask = mask_np
        mask = mask.astype(np.float32)
        if mask.shape != gen_np.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image {gen_np.shape[:2]}")
        mask_3c = np.repeat(mask[..., None], 3, axis=-1)
        gen_m = gen_np * mask_3c
        gt_m = gt_np * mask_3c
        try:
            # For masked PSNR, compute MSE only inside mask
            diff = (gen_m - gt_m) ** 2
            mse = diff.sum() / (mask_3c.sum() * 3 + 1e-8)

            if mse > 0:
                psnr_val = 10 * np.log10(255 ** 2 / mse)
            else:
                psnr_val = float("inf")

        except Exception as e:
            print(f"PSNR failed: {e}")
        try:
            gen_ssim = gen_m.astype(np.uint8)
            gt_ssim = gt_m.astype(np.uint8)

            ssim_val = ssim(
                gen_ssim,
                gt_ssim,
                data_range=255,
                channel_axis=-1,
                gaussian_weights=True,
            )
        except Exception as e:
            print(f"SSIM failed: {e}")
        try:
            gen_t = lpips_transform(gen_pil_rgb).unsqueeze(0).to(device)
            gt_t = lpips_transform(gt_pil_rgb).unsqueeze(0).to(device)

            if model_dtype:
                gen_t = gen_t.to(model_dtype)
                gt_t = gt_t.to(model_dtype)
            # Apply mask: convert mask to torch and resize
            mask_t = torch.from_numpy(mask).float().to(device)[None, None, :, :]
            mask_t = torch.nn.functional.interpolate(mask_t, size=gen_t.shape[2:], mode="bilinear")
            gen_t = gen_t * mask_t
            gt_t = gt_t * mask_t
            lpips_val = lpips_model(gen_t, gt_t).item()
        except Exception as e:
            print(f"LPIPS failed: {e}")
    except Exception as e:
        print(f"General error in metric preprocessing: {e}")
    return psnr_val, ssim_val, lpips_val



@torch.no_grad()
def extract_flare_channels(
    shader: Any,
    gbuffer: Dict[str, torch.Tensor],
    views_subset: Dict[str, Any],
    mesh: Any,
) -> Dict[str, torch.Tensor]:
    """
    Extracts Albedo, Normal, Roughness, and Irradiance from the FLARE NeuralShader
    using the inputs available during the evaluation loop.

    Args:
        shader (NeuralShader): The loaded and evaluated NeuralShader model.
        gbuffer (dict): The g-buffer dictionary populated by the renderer and shader.
                        Requires 'canonical_position' and 'mask'.
        views_subset (dict): The dictionary containing camera/view information.
        mesh (object): The mesh object.

    Returns:
        dict: A dictionary containing the extracted, masked channels:
              {
                  "albedo": (B, H, W, 3),
                  "normal": (B, H, W, 3),
                  "roughness": (B, H, W, 1),
                  "irradiance": (B, H, W, 3),
                  "specular_intensity": (B, H, W, 1)
              }
    """
    position = gbuffer["canonical_position"]  # For material MLP (PE)
    bz, h, w, ch = position.shape
    device = shader.device
    view_direction = torch.cat(
        [v.center.unsqueeze(0) for v in views_subset["camera"]], dim=0
    )
    view_dir = view_direction[:, None, None, :]  # (B, 1, 1, 3)

    pe_input = shader.apply_pe(position=position)
    all_tex = shader.material_mlp(pe_input.view(-1, shader.inp_size).to(torch.float32))

    # kd (Albedo)
    kd = all_tex[..., :3].view(bz, h, w, ch)

    # kr (Roughness)
    kr = all_tex[..., 3:4].view(bz, h, w, 1).to(device)

    # ko (Specular Intensity)
    ko = all_tex[..., 4:5].view(bz, h, w, 1).to(device)
    normal_bend = gbuffer["normal"]

    # Replicate the light_mlp logic from the 'forward' method
    kr_max = torch.ones((bz, h, w, 1)).to(device)

    # Encode normals and roughness (using max roughness)
    enc_nd_kr_max = shader.dir_enc_func(normal_bend.view(-1, 3), kr_max.view(-1, 1))

    # Get diffuse shading from the light MLP
    shading = shader.light_mlp(enc_nd_kr_max)
    irradiance = shading.view(bz, h, w, 3)
    mask = gbuffer["mask"].float()  # (B, H, W, 1)
    albedo_masked = torch.lerp(torch.zeros_like(kd), kd, mask)
    normal_masked = torch.lerp(torch.zeros_like(normal_bend), normal_bend, mask)
    roughness_masked = torch.lerp(torch.zeros_like(kr), kr, mask)
    irradiance_masked = torch.lerp(torch.zeros_like(irradiance), irradiance, mask)
    specular_intensity_masked = torch.lerp(torch.zeros_like(ko), ko, mask)

    return {
        "albedo": albedo_masked,
        "normal": normal_masked,
        "roughness": roughness_masked,
        "irradiance": irradiance_masked,
        "specular_intensity": specular_intensity_masked,
    }
