from pathlib import Path

import PIL
import torch
import torchvision
from diffusers import DDIMScheduler
from rgbx.rgb2x.load_image import load_ldr_image
from rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from scripts.util import save_intrinsic_channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_rgbx_pipeline(model_cache: str) -> StableDiffusionAOVMatEstPipeline:
    """
    Build and return a Stable Diffusion RGB-to-X pipeline with custom scheduler settings.

    Parameters:
        model_cache (str): Directory path to cache or load the pretrained model.

    Returns:
        StableDiffusionAOVMatEstPipeline: Configured pipeline ready for inference.
    """

    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=model_cache,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )

    pipe.set_progress_bar_config(disable=False)

    pipe = pipe.to(device)
    return pipe


def infer_intrinsic_channels(
    image_path: str,
    pipe: StableDiffusionAOVMatEstPipeline,
    inference_steps: int = 50,
    seed: int = 0,
) -> dict[str, PIL.Image]:
    """
    Infer intrinsic channel images from an input image using a Stable Diffusion pipeline.

    Parameters:
        image_path (str): Path to the input image.
        pipe (StableDiffusionAOVMatEstPipeline): Preconfigured RGB-to-X pipeline.
        inference_steps (int, optional): Number of steps for the diffusion inference. Default is 50.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        dict[str, PIL.Image]: Dict mapping channel names
        ('rgb', 'albedo', 'normal', 'roughness', 'metallic', 'irradiance') to their images.
    """

    generator = torch.Generator(device=device).manual_seed(seed)
    photo = load_ldr_image(image_path, from_srgb=True)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    intrinsic_channels = {"rgb": torchvision.transforms.functional.to_pil_image(photo, mode=None)}

    for aov_name in required_aovs:
        prompt = prompts[aov_name]
        generated_image = pipe(
            prompt=prompt,
            photo=photo,
            num_inference_steps=inference_steps,
            height=photo.shape[0],
            width=photo.shape[1],
            generator=generator,
            required_aovs=[aov_name],
        ).images[0][0]

        intrinsic_channels[aov_name] = generated_image

    return intrinsic_channels


def infer_intrinsic_channels_for_dataset(
    dataset_path: str, save_dir: str, model_cache: str
) -> None:
    """
    Run intrinsic channel inference on all images in a dataset and save the results.

    Parameters:
        dataset_path (str): Directory containing input .png images.
        save_dir (str): Directory where the inferred intrinsic channels will be saved.
        model_cache (str): Directory used to cache or load the RGB-to-X model.
    """

    relevant_files = Path(dataset_path).glob("*.png")

    pipe = build_rgbx_pipeline(model_cache)

    for image_path in sorted(relevant_files):
        intrinsic_channels = infer_intrinsic_channels(str(image_path), pipe=pipe)

        name = Path(image_path).stem

        save_intrinsic_channels(intrinsic_channels, name=name, save_dir=save_dir)
