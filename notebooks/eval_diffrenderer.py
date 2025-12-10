import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio
from contextlib import nullcontext
from types import SimpleNamespace
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate import PartialState
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import lpips
import sys
from pathlib import Path
import matplotlib.pyplot as plt 
import pandas as pd
import json

from material_aware_flare.eval import calculate_metrics

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'external', 'diffusion-renderer')
if module_path not in sys.path:
    sys.path.append(module_path)
from external.diffusion_renderer.src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
from external.diffusion_renderer.utils.utils_rgbx import convert_rgba_to_rgb_pil
from external.diffusion_renderer.utils.utils_rgbx_inference import (
    touch, 
    find_images_recursive, 
    base_plus_ext,
    group_images_into_videos, 
    split_list_with_overlap, 
    resize_upscale_without_padding
)


def main():
    eval_transform_lpips = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cfg = SimpleNamespace(
        inference_model_weights="/home/hleonhard/adl4cv_ws25-26_Relightable-Avatars/notebooks/checkpoints/diffusion_renderer-inverse-svd",
        inference_input_dir="/home/hleonhard/data/flare_subject_data/001/MVI_1812/image/",
        inference_save_dir="/home/hleonhard/data//output_diffusionrenderer_001/MVI_1812/",

        # Inference Parameters
        inference_n_frames=2,
        overlap_n_frames=1,
        inference_n_steps=20,
        chunk_mode='all',  # 'first' or 'all'
        model_passes=[ 'roughness', 'normal', 'diffuse_albedo'],
        inference_res=[512, 512],

        # Model Config 
        weight_dtype='fp16',
        cond_mode="skip",
        use_deterministic_mode=False,
        seed=42,
        autocast=True,

        # SVD-specific parameters
        inference_min_guidance_scale=1.0,
        inference_max_guidance_scale=3.0,
        fps=7,
        motion_bucket_id=127,
        cond_aug=0,
        decode_chunk_size=None,

        # Data Loading
        image_group_mode="folder", # 'folder' or 'individual'
        subsample_every_n_frames=1,
        image_extensions=['.png', '.jpg', '.jpeg'],

        # Saving
        save_image=False,
        save_video=False,
        save_video_fps=7,

        # Evaluation
        do_evaluation=True,
        flare_albedo_dir="/home/hleonhard/data/flare_models/001/MVI_1812/images_evaluation/albedo/",
        flare_normal_dir="/home/hleonhard/data/flare_models/001/MVI_1812/images_evaluation/normal/",
        flare_roughness_dir="/home/hleonhard/data/flare_models/001/MVI_1812/images_evaluation/roughness/",
        flare_mask_dir='/home/hleonhard/data/flare_subject_data/001/MVI_1812/mask/',
    )

    # Post-process config (from original code)
    cfg.inference_height, cfg.inference_width = cfg.inference_res
    if cfg.weight_dtype == 'fp16':
        cfg.torch_dtype = torch.float16
    elif cfg.weight_dtype == 'fp32':
        cfg.torch_dtype = torch.float32

    assert cfg.flare_albedo_dir is not None, "flare_albedo_dir must be provided for evaluation"
    assert cfg.flare_normal_dir is not None, "flare_normal_dir must be provided for evaluation"
    assert cfg.flare_roughness_dir is not None, "flare_roughness_dir must be provided for evaluation"
    assert cfg.flare_mask_dir is not None
    assert os.path.isdir(cfg.flare_albedo_dir), f"Flare albedo dir not found: {cfg.flare_albedo_dir}"
    assert os.path.isdir(cfg.flare_normal_dir), f"Flare normal dir not found: {cfg.flare_normal_dir}"
    assert os.path.isdir(cfg.flare_roughness_dir), f"Flare roughness dir not found: {cfg.flare_roughness_dir}"
    assert os.path.isdir(cfg.flare_mask_dir)
    print("Evaluation enabled. Comparing against:")
    print(f"  Albedo: {cfg.flare_albedo_dir}")
    print(f"  Normal: {cfg.flare_normal_dir}")
    print(f"  Roughness: {cfg.flare_roughness_dir}")

    # Setup Accelerator
    accelerator = Accelerator()
    distributed_state = PartialState()
    device = accelerator.device

    print(f"Using device: {device}")

    missing_kwargs = {}
    missing_kwargs["cond_mode"] = cfg.cond_mode
    missing_kwargs["use_deterministic_mode"] = cfg.use_deterministic_mode

    if os.path.exists(cfg.inference_model_weights):
        model_weights_subfolders = os.listdir(cfg.inference_model_weights)
    else:
        model_weights_subfolders = []

    if "image_encoder" not in model_weights_subfolders:
        print("Downloading missing image_encoder from StabilityAI...")
        missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder",
        )
        assert cfg.cond_mode != "image", "Image encoder missing but cond_mode is 'image'"
    if "feature_extractor" not in model_weights_subfolders:
        print("Downloading missing feature_extractor from StabilityAI...")
        missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor",
        )
        assert cfg.cond_mode != "image", "Feature extractor missing but cond_mode is 'image'"

    pipeline = RGBXVideoDiffusionPipeline.from_pretrained(cfg.inference_model_weights, **missing_kwargs)
    pipeline = pipeline.to(device)
    pipeline = pipeline.to(cfg.torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    lpips_model = lpips.LPIPS(net='vgg').to(device).to(cfg.torch_dtype)

    # Diffusion Renderer requires the images to be grouped into sequences
    validation_image_paths = find_images_recursive(
        cfg.inference_input_dir, image_extensions=cfg.image_extensions
    )
    print(f"Found {len(validation_image_paths)} total images.")

    validation_video_list = group_images_into_videos(
        validation_image_paths,
        image_group_mode=cfg.image_group_mode,
        subsample_every_n_frames=cfg.subsample_every_n_frames,
    )
    print(f"Grouped into {len(validation_video_list)} videos/sequences.")

    os.makedirs(cfg.inference_save_dir, exist_ok=True)
    success_signal_dir = os.path.join(cfg.inference_save_dir, "TMP_SUCCESS_SIGNAL")
    os.makedirs(success_signal_dir, exist_ok=True)

    if os.path.exists(success_signal_dir):
        filtered_validation_video_list = []
        for input_image_relative_path_list in validation_video_list:
            video_relative_base_name = base_plus_ext(
                input_image_relative_path_list[0], mode=cfg.image_group_mode
            )[0]
            input_image_relative_path_chunks = split_list_with_overlap(
                input_image_relative_path_list,
                cfg.inference_n_frames,
                cfg.overlap_n_frames,
                chunk_mode=cfg.chunk_mode,
            )
            if not input_image_relative_path_chunks:
                continue

            max_chunk_ind = len(input_image_relative_path_chunks) - 1
            success_signal_str = (
                video_relative_base_name.replace("/", "--") + f".{max_chunk_ind:04d}"
            )
            success_signal_path = os.path.join(success_signal_dir, success_signal_str)

            if os.path.exists(success_signal_path):
                print(f"Skipping already processed video: {success_signal_str}")
            else:
                filtered_validation_video_list.append(input_image_relative_path_list)

        validation_video_list = filtered_validation_video_list
        print(f"{len(validation_video_list)} videos remaining to process.")

    processing_list = validation_video_list
    marker = pd.read_csv("./notebooks/diffusionrenderer_001_mvi_1812_subset.csv", index_col=0)

    all_metrics_data = {'diffuse_albedo': [], 'normal': [], 'roughness': []}

    if 'lpips_model' in locals() or 'lpips_model' in globals():
        lpips_model = lpips_model.to(device, dtype=torch.float32)
        lpips_model.eval()

    print("Starting inference and evaluation...")
    for i, input_image_relative_path_list in tqdm(
        enumerate(processing_list), desc="Processing Videos"
    ):
        video_relative_base_name = base_plus_ext(
            input_image_relative_path_list[0], mode=cfg.image_group_mode
        )[0]
        video_relative_base_name = '/home/hleonhard/data/diff_renderer_001_MVI_1812'
        os.makedirs(
                video_relative_base_name,
                exist_ok=True,
            )
        print(f"Images will be saved to {video_relative_base_name}")

        # Split into chunks
        input_image_relative_path_chunks = split_list_with_overlap(
            input_image_relative_path_list,
            cfg.inference_n_frames,
            cfg.overlap_n_frames,
            chunk_mode=cfg.chunk_mode,
        )
        if len(input_image_relative_path_chunks) == 0:
            continue

        if cfg.save_image:
            os.makedirs(
                os.path.join(cfg.inference_save_dir, f"{video_relative_base_name}"),
                exist_ok=True,
            )
        

        for chunk_ind in tqdm(
            range(0, len(input_image_relative_path_chunks), 2),
            desc="  Processing Chunks",
            leave=False,
        ):
            success_signal_str = (
                video_relative_base_name.replace("/", "--") + f".{chunk_ind:04d}"
            )
            success_signal_path = os.path.join(success_signal_dir, success_signal_str)
            if os.path.exists(success_signal_path):
                print(f"Skipping chunk: {success_signal_str}")
                continue

            current_image_relative_path_list = input_image_relative_path_chunks[chunk_ind]
            # check if we skip this index
            if current_image_relative_path_list[0] in marker.iloc[:,0].values:
                # already saw this 
                print(f"Skipping chunk: {current_image_relative_path_list[0]}")
                continue
            # Fill frames to inference_n_frames
            while len(current_image_relative_path_list) < cfg.inference_n_frames:
                current_image_relative_path_list.append(
                    current_image_relative_path_list[-1]
                )

            # Process input image
            input_images_uint8 = []
            for ind in range(cfg.inference_n_frames):
                input_path = os.path.join(
                    cfg.inference_input_dir, current_image_relative_path_list[ind]
                )
                input_image_pil = Image.open(input_path)
                input_image_pil = convert_rgba_to_rgb_pil(
                    input_image_pil, background_color=(0, 0, 0)
                )

                if ind == 0:
                    width, height = input_image_pil.size
                    if width != cfg.inference_width or height != cfg.inference_height:
                        input_image_pil = resize_upscale_without_padding(
                            input_image_pil, cfg.inference_height, cfg.inference_width
                        )
                        width, height = input_image_pil.size
                else:
                    if (
                        width != input_image_pil.size[0]
                        or height != input_image_pil.size[1]
                    ):
                        input_image_pil = input_image_pil.resize(
                            (width, height), resample=Image.BILINEAR
                        )

                if cfg.save_image:
                    save_path = os.path.join(
                        cfg.inference_save_dir,
                        f"{video_relative_base_name}/{chunk_ind:04d}.{ind:04d}.rgb.png",
                    )
                    input_image_pil.save(save_path)

                input_images_uint8.append(np.asarray(input_image_pil))

            # Formatting input
            input_images = (
                np.stack(input_images_uint8, axis=0)[None, ...].astype(np.float32) / 255.0
            )  # (1, F, H, W, C)
            cond_images = {"rgb": input_images}
            cond_labels = {"rgb": "vae"}
            if cfg.cond_mode == "image":
                cond_images["clip_img"] = input_images[
                    :, 0:1, ...
                ]  # NOTE: clip uses first frame only
                cond_labels["clip_img"] = "clip"

            viz_images_uint8 = input_images_uint8
            for inference_pass in cfg.model_passes:
                cond_images["input_context"] = inference_pass

                # DiffusionRenderer Pipeline
                generator = None
                if cfg.seed is not None:
                    generator = torch.Generator(device=device).manual_seed(cfg.seed)

                autocast_ctx = (
                    torch.autocast(device.type, enabled=cfg.autocast)
                    if not torch.backends.mps.is_available()
                    else nullcontext()
                )

                with autocast_ctx:
                    inference_image_list = pipeline(
                        cond_images,
                        cond_labels,
                        height=height,
                        width=width,
                        num_frames=cfg.inference_n_frames,
                        num_inference_steps=cfg.inference_n_steps,
                        min_guidance_scale=cfg.inference_min_guidance_scale,
                        max_guidance_scale=cfg.inference_max_guidance_scale,
                        fps=cfg.fps,
                        motion_bucket_id=cfg.motion_bucket_id,
                        noise_aug_strength=cfg.cond_aug,
                        generator=generator,
                        decode_chunk_size=cfg.decode_chunk_size,
                    ).frames[0]

                # RGBX Pipeline

                # Save images and run evaluation
                for ind in range(len(inference_image_list)):
                    if cfg.save_image or True:
                        # save images for jonathan
                        input_image_filename = os.path.basename(
                            current_image_relative_path_list[ind]
                        )
                        save_path = os.path.join(
                            cfg.inference_save_dir,
                            f"{video_relative_base_name}/{input_image_filename}_{inference_pass}.png",
                        )
                        inference_image_list[ind].save(save_path)
                    # --- EVALUATION LOGIC ---
                    if cfg.do_evaluation and inference_pass in [
                        "diffuse_albedo",
                        "normal",
                        "roughness"
                    ]:
                        gt_image_pil = inference_image_list[ind]
                        input_image_filename = os.path.basename(
                            current_image_relative_path_list[ind]
                        )
                        gt_path = None
                        if inference_pass == "diffuse_albedo":
                            gt_path = os.path.join(
                                cfg.flare_albedo_dir, input_image_filename.zfill(8)
                            )
                        elif inference_pass == "normal":
                            gt_path = os.path.join(
                                cfg.flare_normal_dir, input_image_filename.zfill(8)
                            )
                        elif inference_pass == "roughness":
                            gt_path = os.path.join(
                                cfg.flare_roughness_dir, input_image_filename.zfill(8)
                            )
                        mask_path = os.path.join(
                                cfg.flare_mask_dir, input_image_filename
                            )
                        if gt_path and os.path.exists(gt_path):
                            gen_image_pil = Image.open(gt_path)
                            mask = plt.imread(mask_path)
                            psnr_val, ssim_val, lpips_val = calculate_metrics(
                                gen_pil=gen_image_pil,
                                gt_pil=gt_image_pil,
                                lpips_model=lpips_model,
                                lpips_transform=eval_transform_lpips,
                                device=device,
                                mask_np=mask,
                                model_dtype=cfg.torch_dtype,
                            )
                            metrics_data = {
                                "file": input_image_filename,
                                "pass": inference_pass,
                                "psnr": psnr_val,
                                "ssim": ssim_val,
                                "lpips": lpips_val,
                            }
                            print(f"created data {metrics_data}")
                            all_metrics_data[inference_pass].append(metrics_data)
                        elif gt_path:
                            print(f"  [Eval] GT file not found, skipping: {gt_path}")
                if cfg.save_video:
                    for ind in range(len(viz_images_uint8)):
                        viz_images_uint8[ind] = np.concatenate(
                            [
                                viz_images_uint8[ind],
                                np.asarray(inference_image_list[ind]),
                            ],
                            axis=1,
                        )

            if cfg.save_video:
                save_path = os.path.join(
                    cfg.inference_save_dir,
                    f"{video_relative_base_name}.{chunk_ind:04d}.viz.mp4",
                )
                imageio.mimsave(
                    save_path, viz_images_uint8, fps=cfg.save_video_fps, codec="h264"
                )
    with open("diff_renderer_res.json", "w") as file:
        json.dump(all_metrics_data, file)
    passes = ['diffuse_albedo', 'normal', 'roughness']
    print("\n" + "="*30)
    print(" FINAL EVALUATION METRICS ON 001")
    print("="*30)
    # In a single-node script, no 'gather' is needed.
    all_metrics_list = all_metrics_data['diffuse_albedo'] + all_metrics_data['normal']  + all_metrics_data['roughness']
    
    if len(all_metrics_list) == 0:
        print("No metrics gathered.")
    else:
        df = pd.DataFrame(all_metrics_list)
        # Calculate and print averages
        for pass_name in passes:
            pass_df = df[df['pass'] == pass_name]
            if pass_df.empty:
                print(f"\nNo metrics calculated for {pass_name}.")
                continue
            avg_psnr = pass_df['psnr'].mean()
            avg_ssim = pass_df['ssim'].mean()
            avg_lpips = pass_df['lpips'].mean()
            print(f"\n--- Average Metrics for: {pass_name} ---")
            print(f"  PSNR:  {avg_psnr:.4f}")
            print(f"  SSIM:  {avg_ssim:.4f}")
            print(f"  LPIPS: {avg_lpips:.4f}")

if __name__ == '__main__':
    main()