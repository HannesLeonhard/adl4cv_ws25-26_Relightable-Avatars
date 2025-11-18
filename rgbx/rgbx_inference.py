import PIL
import torch
import torchvision
from diffusers import DDIMScheduler
from external.rgbx.rgb2x.load_image import load_ldr_image
from external.rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_rgbx_pipeline(model_cache: str) -> StableDiffusionAOVMatEstPipeline:
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
    generator = torch.Generator(device=device).manual_seed(seed)
    photo = load_ldr_image(image_path)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    intrinsic_channels = {
        "rgb": torchvision.transforms.functional.to_pil_image(photo, mode=None)
    }

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
