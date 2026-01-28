# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import json
import numpy as np
from pathlib import Path
from gpytoolbox import remesh_botsch
import torch
from tqdm import tqdm
from flame.FLAME import FLAME
from flare.dataset import *
from flare.dataset import dataset_util
from flare.core import Mesh, Renderer
from flare.losses import *
from flare.modules import NeuralShader, get_deformer_network, Displacement
from flare.utils import AABB, read_mesh, write_mesh
import nvdiffrec.render.light as light

from scripts.config import PathConfig, MaterialAwareTrainingConfig
from scripts.material_diffusion_regularization import (
    diffusion_albedo_regularization,
    diffusion_irradiance_regularization,
    diffusion_normal_regularization,
    diffusion_roughness_regularization,
)

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(path_config: PathConfig, args: MaterialAwareTrainingConfig, dataset_train, dataloader_train, FLAMEServer, diff_reg_decay_schedule = None):
    ## ============== Dir ==============================
    meshes_save_path = path_config.meshes_save_path(args.stage)
    shaders_save_path = path_config.shaders_save_path(args.stage)

    ## ============== load mesh/train mesh ==============================
    if args.finetune_color:
        mesh_path = path_config.experiment_dir / "stage_1" / "meshes" / f"mesh_latest.obj"
        print("loading mesh from:", mesh_path)
        flame_canonical_mesh = read_mesh(mesh_path, device=device)
        flame_canonical_mesh.compute_connectivity()
        flame_canonical_mesh.to(device)
    else:
        if args.downsample:
            v_down, f_down = remesh_botsch(FLAMEServer.canonical_verts.squeeze(0).cpu().detach().numpy().astype(np.float64), 
                                                                    FLAMEServer.faces_tensor.cpu().numpy().astype(np.int32), h=float(args.downsample_ratio))
            verts = np.ascontiguousarray(v_down)
            faces = np.ascontiguousarray(f_down)
            print("Downsampled:", verts.shape, faces.shape)
        else:
            verts = FLAMEServer.canonical_verts.squeeze(0)
            faces = FLAMEServer.faces_tensor

        flame_canonical_mesh: Mesh = None
        flame_canonical_mesh = Mesh(verts, faces, device=device)
        flame_canonical_mesh.compute_connectivity()
        write_mesh(Path(meshes_save_path / "init_mesh.obj"), flame_canonical_mesh.to('cpu'))

    ## ============== renderer ==============================
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [torch.min(flame_canonical_mesh.vertices, dim=0).values, torch.max(flame_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # ==============================================================================================
    # vertices
    # ==============================================================================================

    lr_vertices = args.lr_vertices
    displacements = Displacement(vertices_shape=flame_canonical_mesh.vertices.shape)
    
    displacements.to(device=device)
    optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=lr_vertices)

    # ==============================================================================================
    # deformation 
    # ==============================================================================================
    if args.train_deformer:
        model_path = None
        print("=="*50)
        print("Training Deformer")
    else:
        print("=="*50)
        print("Loading deformer network trained in the previous stage")
        args.weight_flame_regularization = 0.0

        model_path = Path(path_config.experiment_dir / "stage_1" / "network_weights" / f"deformer_latest.pt")
        assert os.path.exists(model_path)

    deformer_net = get_deformer_network(FLAMEServer, model_path=model_path, train=args.train_deformer, d_in=3, dims=args.deform_dims, 
                                           weight_norm=True, multires=0, num_exp=50, aabb=flame_mesh_aabb, ghostbone=args.ghostbone, device=device)
    if args.train_deformer:
        optimizer_deformer = torch.optim.Adam(list(deformer_net.parameters()), lr=args.lr_deformer)

    # ==============================================================================================
    # shading
    # ==============================================================================================

    lgt = light.create_env_rnd()    
    disentangle_network_params = {
        "material_mlp_ch": args.material_mlp_ch,
        "light_mlp_ch":args.light_mlp_ch,
        "material_mlp_dims":args.material_mlp_dims,
        "light_mlp_dims":args.light_mlp_dims
    }

    # Create the optimizer for the neural shader
    shader = NeuralShader(fourier_features=args.fourier_features,
                          activation=args.activation,
                          last_activation=torch.nn.Sigmoid(), 
                          disentangle_network_params=disentangle_network_params,
                          bsdf=args.bsdf,
                          aabb=flame_mesh_aabb,
                          device=device)
    params = list(shader.parameters()) 

    if args.weight_albedo_regularization > 0:
        from robust_loss_pytorch.adaptive import AdaptiveLossFunction
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters()) ## need to train it

    optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader)

    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "normal": args.weight_normal,
        "laplacian": args.weight_laplacian,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "albedo_regularization": args.weight_albedo_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "white_light_regularization": args.weight_white_lgt_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff,
        "diffusion_normal": args.weight_diffusion_normal_regularization,
        "diffusion_albedo": args.weight_diffusion_albedo_regularization,
        "diffusion_roughness": args.weight_diffusion_roughness_regularization,
        "diffusion_irradiance": args.weight_diffusion_irradiance_regularization,
        "flame_regularization" : 1.0 if args.train_deformer else 0.0
    }

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)
    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    print("=="*50)
    shader.train()
    if args.train_deformer:
        deformer_net.train()
    displacements.train()
    print("Batch Size:", args.batch_size)
    print("=="*50)

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================
    diffusion_normal_losses = []
    diffusion_albedo_losses = []
    diffusion_roughness_losses = []
    diffusion_irradiance_losses = []

    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0

    iterations = epochs * len(dataloader_train)
    print("Iterations:", iterations)

    progress_bar = tqdm(range(epochs))
    start = time.time()
    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')
            
            # ==============================================================================================
            # upsample + remesh + reduce lr + freeze if required
            # ==============================================================================================
            if iteration in args.upsample_iterations and not args.finetune_color:
                print("=="*50)
                print("Upsampling at iteration:", iteration)
                # Upsample the mesh by remeshing the surface with half the average edge length
                e0, e1 = mesh.edges.unbind(1)

                average_edge_length = torch.linalg.norm(canonical_offset_vertices[e0] - canonical_offset_vertices[e1], dim=-1).mean()
                v_upsampled, f_upsampled = remesh_botsch(canonical_offset_vertices.cpu().detach().numpy().astype(np.float64), 
                                                        mesh.indices.cpu().numpy().astype(np.int32), h=float(average_edge_length/1.5))
                v_upsampled = np.ascontiguousarray(v_upsampled)
                f_upsampled = np.ascontiguousarray(f_upsampled)
                flame_canonical_mesh = Mesh(v_upsampled, f_upsampled, device=device)
                flame_canonical_mesh.compute_connectivity()

                print("Vertices:", v_upsampled.shape)
                print("Faces:", f_upsampled.shape)
                del v_upsampled, f_upsampled
                if iteration == args.upsample_iterations[0]:
                    lr_vertices *= 0.75
                    # Adjust weights and step size
                    loss_weights['laplacian'] *= 4
                    loss_weights['normal'] *= 4
                print("laplacian weight", loss_weights['laplacian'])
                print("normal consistency weight", loss_weights['normal'])
                print("lr vertices", lr_vertices)

                displacements.register_parameter('vertex_offsets', torch.nn.Parameter(torch.zeros(flame_canonical_mesh.vertices.shape), requires_grad=True))
                displacements.canonical_vertices = flame_canonical_mesh.vertices
                displacements.vertices_shape = flame_canonical_mesh.vertices.shape
                displacements.to(device=device)
                optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=lr_vertices)
                print("=="*50)

            # ==============================================================================================
            # update/displace vertices
            # ==============================================================================================
            v_off = displacements()
            canonical_offset_vertices = flame_canonical_mesh.vertices + v_off
            mesh = flame_canonical_mesh.with_vertices(canonical_offset_vertices)

            # ==============================================================================================
            # deformation of canonical mesh
            # ==============================================================================================      
            shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
            
            batched_verts = mesh.vertices.unsqueeze(0).repeat(args.batch_size, 1, 1)
            _, pose_features, transformations = FLAMEServer(expression_params=views_subset["flame_expression"], full_pose=views_subset["flame_pose"])
            if args.ghostbone:
                transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(args.batch_size, -1, -1, -1).float().to(device), transformations], 1)
            deformed_vertices = FLAMEServer.forward_pts_batch(pnts_c=batched_verts, betas=views_subset["flame_expression"], transformations=transformations, pose_feature=pose_features, 
                                                shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights, dtype=torch.float32, map2_flame_original=True)
            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

            # ==============================================================================================
            # R A S T E R I Z A T I O N
            # ==============================================================================================
            gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices) 
            
            # ==============================================================================================
            # loss function 
            # ==============================================================================================
            ## ============== geometry regularization ==============================
            losses['normal'] = normal_consistency_loss(mesh)
            losses['laplacian'] = laplacian_loss(mesh)

            ## ============== color + regularization for color ==============================
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, args.batch_size)
            losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            
            losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)

            ## ======= regularization color ========
            losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, displacements, iteration)
            losses['white_light_regularization'] = white_light(cbuffers)
            losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=args.r_mean)
            losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
            
            ## ============== flame regularization ==============================
            if loss_weights['flame_regularization'] > 0:
                losses['flame_regularization'], gt_nn = flame_regularization(FLAMEServer, lbs_weights, shapedirs, posedirs, mesh.vertices, args.ghostbone, 
                                                                      iteration, args.flame_mask, views_subset=views_subset, gbuffer=gbuffers, 
                                                                      weight_lbs=args.weight_flame_regularization)
            
                if iteration in args.decay_flame:
                    print("Decaying flame regularization")
                    loss_weights['flame_regularization'] *= 0.5


            if diff_reg_decay_schedule is not None:
                diff_reg_decay_schedule(loss_weights, iteration, iterations, args)


            losses["diffusion_normal"] = diffusion_normal_regularization(
                gbuffers["normal"],
                views_subset["diffusion_normal"],
                views_subset["skin_mask"],
                views_subset["mask"],
            )
            losses["diffusion_albedo"] = diffusion_albedo_regularization(
                cbuffers["albedo"],
                views_subset["diffusion_albedo"],
                views_subset["skin_mask"],
                views_subset["mask"],
            )
            losses["diffusion_roughness"] = diffusion_roughness_regularization(
                cbuffers["roughness"],
                views_subset["diffusion_roughness"],
                views_subset["skin_mask"],
                views_subset["mask"],
            )
            losses["diffusion_irradiance"] = diffusion_irradiance_regularization(
                cbuffers["irradiance"],
                views_subset["diffusion_irradiance"],
                views_subset["skin_mask"],
                views_subset["mask"],
            )

            diffusion_normal_losses.append(losses["diffusion_normal"].item())
            diffusion_albedo_losses.append(losses["diffusion_albedo"].item())
            diffusion_roughness_losses.append(losses["diffusion_roughness"].item())
            diffusion_irradiance_losses.append(losses["diffusion_irradiance"].item())

            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                loss += v * loss_weights[k]

            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            optimizer_shader.zero_grad()
            optimizer_vertices.zero_grad()
            if args.train_deformer:
                optimizer_deformer.zero_grad()

            loss.backward()
            torch.cuda.synchronize()

            ### increase the gradients of positional encoding following tinycudnn
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0

            optimizer_shader.step()
            optimizer_vertices.step()
            if args.train_deformer:
                optimizer_deformer.step()

            progress_bar.set_postfix({'loss': loss.detach().cpu().item()})

            # ==============================================================================================
            # warning: check if light mlp diverged
            # ==============================================================================================
            '''
            We do not use an activation function for the output layer of light MLP because we are learning in sRGB space where the values 
            are not restricted between 0 and 1. As a result, the light MLP diverges sometimes and predicts only zero values. 
            Hence, we have included the try and catch block to automatically restart the training during this case. 
            '''
            if iteration == 100:
                convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)
                try:
                    diffuse_shading = convert_uint(cbuffers["shading"])
                    specular_shading = convert_uint(cbuffers["specu"])
                    if torch.count_nonzero(diffuse_shading) == 0 or torch.count_nonzero(specular_shading) == 0:
                        raise ValueError("All values predicted from light MLP are zero")
                except ValueError as e:
                    print(f"Error: {e}")
                    raise  # Raise the exception to exit the current execution of main()

    end = time.time()
    total_time = ((end - start) % 3600)
    print("TIME TAKEN (mins):", int(total_time // 60))
    # ==============================================================================================
    # s a v e
    # ==============================================================================================
    with open(path_config.experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)
    write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.detach().to('cpu'))
    shader.save(shaders_save_path / f'shader_latest.pt')
    displacements.save(shaders_save_path / f'displacement_latest.pt')
    deformer_net.save(shaders_save_path / f'deformer_latest.pt')

    return diffusion_normal_losses, diffusion_albedo_losses, diffusion_roughness_losses, diffusion_irradiance_losses


def material_aware_training(
    path_config: PathConfig,
    args1: MaterialAwareTrainingConfig,
    args2: MaterialAwareTrainingConfig,
    dataset_train: DatasetLoader,
    dataloader_train,
    diff_reg_decay_schedule = None
):
    ## ============== load FLAME mesh ==============================
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(path_config.flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)

    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if args1.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)

    diffusion_normal_losses, diffusion_albedo_losses, diffusion_roughness_losses, diffusion_irradiance_losses = [], [], [], []

    while True:
        try:
            n, a, r, i = main(path_config=path_config, args=args1, dataset_train=dataset_train, dataloader_train=dataloader_train, FLAMEServer=FLAMEServer, diff_reg_decay_schedule=diff_reg_decay_schedule)
            diffusion_normal_losses.extend(n)
            diffusion_albedo_losses.extend(a)
            diffusion_roughness_losses.extend(r)
            diffusion_irradiance_losses.extend(i)
            break
        except Exception as exc:
            print("--"*50)
            print(exc)
            print("Warning: Re-initializing main() because the training of light MLP diverged and all the values are zero. If the training does not restart, please end it and restart. ")
            print("--"*50)

    while True:
        try:
            n, a, r, i = main(path_config=path_config, args=args2, dataset_train=dataset_train, dataloader_train=dataloader_train, FLAMEServer=FLAMEServer, diff_reg_decay_schedule=diff_reg_decay_schedule)
            diffusion_normal_losses.extend(n)
            diffusion_albedo_losses.extend(a)
            diffusion_roughness_losses.extend(r)
            diffusion_irradiance_losses.extend(i)
            break
        except Exception as exc:
            print("--"*50)
            print(exc)
            print("Warning: Re-initializing main() because the training of light MLP diverged and all the values are zero. If the training does not restart, please end it and restart. ")
            print("--"*50)

    with open(path_config.experiment_dir / "diffusion_regularization_losses.json", "w") as file:
        json.dump(
            {
                "diffusion_normal_losses" : diffusion_normal_losses,
                "diffusion_albedo_losses" : diffusion_albedo_losses,
                "diffusion_roughness_losses" : diffusion_roughness_losses,
                "diffusion_irradiance_losses" : diffusion_irradiance_losses,
            },
            file,
            indent=2,
        )

    return diffusion_normal_losses, diffusion_albedo_losses, diffusion_roughness_losses, diffusion_irradiance_losses