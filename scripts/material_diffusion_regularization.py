import torch
import torch.nn as nn


def diffusion_albedo_regularization(albedo, albedo_diffusion, semantic, mask):
    skin_mask = (torch.sum(semantic[..., :3], axis=-1)).unsqueeze(-1).expand(albedo.shape)
    skin_mask = skin_mask * mask.expand(albedo.shape)
    mask = (skin_mask > 0.0).int().bool()
    albedo_skin = albedo[mask]
    albedo_diffusion_skin = albedo_diffusion[mask]
    return nn.L1Loss().forward(albedo_skin, albedo_diffusion_skin)


def diffusion_roughness_regularization(roughness, roughness_diffusion, semantic, mask):
    roughness_diffusion = roughness_diffusion[:, :, :, 0].unsqueeze(-1)
    skin_mask = (torch.sum(semantic[..., :3], axis=-1)).unsqueeze(-1)
    skin_mask = skin_mask * mask
    mask = (skin_mask > 0.0).int().bool()
    roughness_skin = roughness[mask]
    roughness_diffusion_skin = roughness_diffusion[mask]
    return nn.L1Loss().forward(roughness_skin, roughness_diffusion_skin)


def diffusion_normal_regularization(normal, normal_diffusion, semantic, mask):
    skin_mask = (torch.sum(semantic[..., :3], axis=-1)).unsqueeze(-1).expand(normal.shape)
    skin_mask = skin_mask * mask.expand(normal.shape)
    mask = (skin_mask > 0.0).int().bool()
    normal_skin = normal[mask]
    normal_diffusion_skin = normal_diffusion[mask]
    return nn.L1Loss().forward(normal_skin, normal_diffusion_skin)


def diffusion_irradiance_regularization(irradiance, irradiance_diffusion, semantic, mask):
    skin_mask = (torch.sum(semantic[..., :3], axis=-1)).unsqueeze(-1).expand(irradiance.shape)
    skin_mask = skin_mask * mask.expand(irradiance.shape)
    mask = (skin_mask > 0.0).int().bool()
    irradiance_skin = irradiance[mask]
    irradiance_diffusion_skin = irradiance_diffusion[mask]
    return nn.L1Loss().forward(irradiance_skin, irradiance_diffusion_skin)
