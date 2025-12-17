from flame.FLAME import FLAME
from flare.core import Mesh, Renderer
from flare.losses import *
from flare.modules import NeuralShader, get_deformer_network, Displacement, ForwardDeformer
from flare.utils import (
    AABB,
    read_mesh,
    visualize_training,
    write_mesh,
)
import nvdiffrec.render.light as light
from flare.dataset import DatasetLoader, dataset_util
from flare.dataset import *
import nvdiffrec.render.light as light
import torch
from pathlib import Path
import numpy as np
from gpytoolbox import remesh_botsch
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
import time
from tqdm import tqdm
from scripts.config import PathConfig, MaterialAwareTrainingConfig, write_config_to_json
from typing import Any
from material_aware_flare.material_diffusion_regularization import (
    diffusion_albedo_regularization, 
    diffusion_irradiance_regularization, 
    diffusion_normal_regularization, 
    diffusion_roughness_regularization
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_flame_server(dataset_train: DatasetLoader, flame_path: str, ghostbone: bool) -> FLAME:
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(Path(flame_path), n_shape=100, n_exp=50, shape_params=flame_shape).to(
        device
    )
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    (
        FLAMEServer.canonical_verts,
        FLAMEServer.canonical_pose_feature,
        FLAMEServer.canonical_transformations,
    ) = FLAMEServer(
        expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose
    )
    if ghostbone:
        FLAMEServer.canonical_transformations = torch.cat(
            [
                torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device),
                FLAMEServer.canonical_transformations,
            ],
            1,
        )
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)

    return FLAMEServer


def build_flare_training_pipeline(
    path_config: PathConfig,
    training_config: MaterialAwareTrainingConfig,
    dataset_train: DatasetLoader,
    FLAMEServer: FLAME,
    stage: str,
):
    # mesh
    if training_config.finetune_color:
        mesh_path = path_config.experiment_dir / "stage_1" / "meshes" / "mesh_latest.obj"
        print("loading mesh from:", mesh_path)
        flame_canonical_mesh = read_mesh(mesh_path, device=device)
        flame_canonical_mesh.compute_connectivity()
        flame_canonical_mesh.to(device)

    else:
        if training_config.downsample:
            v_down, f_down = remesh_botsch(
                FLAMEServer.canonical_verts.squeeze(0).cpu().detach().numpy().astype(np.float64),
                FLAMEServer.faces_tensor.cpu().numpy().astype(np.int32),
                h=float(training_config.downsample_ratio),
            )
            verts = np.ascontiguousarray(v_down)
            faces = np.ascontiguousarray(f_down)
            print("Downsampled:", verts.shape, faces.shape)
        else:
            verts = FLAMEServer.canonical_verts.squeeze(0)
            faces = FLAMEServer.faces_tensor

        flame_canonical_mesh = Mesh(verts, faces, device=device)
        flame_canonical_mesh.compute_connectivity()
        write_mesh(
            path_config.meshes_save_path(stage) / "init_mesh.obj", flame_canonical_mesh.to("cpu")
        )

    # renderer
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [
        torch.min(flame_canonical_mesh.vertices, dim=0).values,
        torch.max(flame_canonical_mesh.vertices, dim=0).values,
    ]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ["mask", "position", "normal", "canonical_position"]
    print("Rasterizing:", channels_gbuffer)

    # displacements
    displacements = Displacement(vertices_shape=flame_canonical_mesh.vertices.shape)
    displacements.to(device=device)

    # deformer
    if training_config.train_deformer:
        model_path = None
        print("Training Deformer")
    else:
        print("Loading deformer network trained in the previous stage")
        model_path = (
            path_config.experiment_dir / "stage_1" / "network_weights" / "deformer_latest.pt"
        )
        assert model_path.exists()

    deformer_net = get_deformer_network(
        FLAMEServer,
        model_path=model_path,
        train=training_config.train_deformer,
        d_in=3,
        dims=training_config.deform_dims,
        weight_norm=True,
        multires=0,
        num_exp=50,
        aabb=flame_mesh_aabb,
        ghostbone=training_config.ghostbone,
        device=device,
    )

    # shader
    lgt = light.create_env_rnd()
    disentangle_network_params = {
        "material_mlp_ch": training_config.material_mlp_ch,
        "light_mlp_ch": training_config.light_mlp_ch,
        "material_mlp_dims": training_config.material_mlp_dims,
        "light_mlp_dims": training_config.light_mlp_dims,
    }

    # Create the optimizer for the neural shader
    shader = NeuralShader(
        fourier_features=training_config.fourier_features,
        activation=training_config.activation,
        last_activation=torch.nn.Sigmoid(),
        disentangle_network_params=disentangle_network_params,
        bsdf=training_config.bsdf,
        aabb=flame_mesh_aabb,
        device=device,
    )

    return (
        flame_canonical_mesh,
        renderer,
        channels_gbuffer,
        displacements,
        deformer_net,
        shader,
        lgt,
    )

def run(args, mesh, views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt):
    ## ============== deform ==============================     
    shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
    eval_vertices = mesh.vertices
    batched_verts = eval_vertices.unsqueeze(0).repeat(views["img"].shape[0], 1, 1)

    _, pose_features, transformations = FLAMEServer(expression_params=views["flame_expression"], full_pose=views["flame_pose"])
    if args.ghostbone:
        transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(views["img"].shape[0], -1, -1, -1).float().to(device), transformations], 1)
    deformed_vertices = FLAMEServer.forward_pts_batch(pnts_c=batched_verts, betas=views["flame_expression"], transformations=transformations, pose_feature=pose_features, 
                                        shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights, dtype=torch.float32, map2_flame_original=True)

    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices)
    
    ## ============== predict color ==============================
    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, args.finetune_color, lgt)

    return rgb_pred, gbuffers, cbuffers


def _material_aware_training(
    path_config: PathConfig,
    training_config: MaterialAwareTrainingConfig,
    dataset_train: DatasetLoader,
    dataloader_train: torch.utils.data.DataLoader,
    FLAMEServer: FLAME,
    stage: str,
    debug_views: Any
):
    flame_canonical_mesh, renderer, channels_gbuffer, displacements, deformer_net, shader, lgt = (
        build_flare_training_pipeline(
            path_config,
            training_config,
            dataset_train,
            FLAMEServer,
            stage,
        )
    )

    if training_config.train_deformer:
        weight_flame_regularization = 1.0
    else:
        weight_flame_regularization = 0.0

    # optimizer
    optimizer_vertices = torch.optim.Adam(
        list(displacements.parameters()), lr=training_config.lr_vertices
    )
    if training_config.train_deformer:
        optimizer_deformer = torch.optim.Adam(
            list(deformer_net.parameters()), lr=training_config.lr_deformer
        )

    params = list(shader.parameters())
    if training_config.weight_albedo_regularization > 0:
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters())
    optimizer_shader = torch.optim.Adam(params, lr=training_config.lr_shader)

    # loss functions
    loss_weights = {
        "mask": training_config.weight_mask,
        "normal": training_config.weight_normal,
        "laplacian": training_config.weight_laplacian,
        "shading": training_config.weight_shading,
        "perceptual_loss": training_config.weight_perceptual_loss,
        "albedo_regularization": training_config.weight_albedo_regularization,
        "roughness_regularization": training_config.weight_roughness_regularization,
        "white_light_regularization": training_config.weight_white_lgt_regularization,
        "fresnel_coeff": training_config.weight_fresnel_coeff,
        "flame_regularization": 1.0 if training_config.train_deformer else 0.0,
        "diffusion_normal": training_config.weight_diffusion_normal_regularization,
        "diffusion_albedo": training_config.weight_diffusion_albedo_regularization,
        "diffusion_roughness": training_config.weight_diffusion_roughness_regularization,
        "diffusion_irradiance": training_config.weight_diffusion_irradiance_regularization,
    }

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)
    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    shader.train()
    if training_config.train_deformer:
        deformer_net.train()
    displacements.train()

    epochs = (training_config.iterations // len(dataloader_train)) + 1
    iteration = 0

    progress_bar = tqdm(range(epochs))
    start = time.time()

    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            iteration += 1
            progress_bar.set_description(desc=f"Epoch {epoch}, Iter {iteration}")

            if (
                iteration in training_config.upsample_iterations
                and not training_config.finetune_color
            ):
                print("Upsampling at iteration:", iteration)

                e0, e1 = mesh.edges.unbind(1)

                average_edge_length = torch.linalg.norm(
                    canonical_offset_vertices[e0] - canonical_offset_vertices[e1], dim=-1
                ).mean()
                v_upsampled, f_upsampled = remesh_botsch(
                    canonical_offset_vertices.cpu().detach().numpy().astype(np.float64),
                    mesh.indices.cpu().numpy().astype(np.int32),
                    h=float(average_edge_length / 1.5),
                )
                v_upsampled = np.ascontiguousarray(v_upsampled)
                f_upsampled = np.ascontiguousarray(f_upsampled)
                flame_canonical_mesh = Mesh(v_upsampled, f_upsampled, device=device)
                flame_canonical_mesh.compute_connectivity()

                print("Vertices:", v_upsampled.shape)
                print("Faces:", f_upsampled.shape)
                del v_upsampled, f_upsampled
                if iteration == training_config.upsample_iterations[0]:
                    training_config.lr_vertices *= 0.75
                    # Adjust weights and step size
                    loss_weights["laplacian"] *= 4
                    loss_weights["normal"] *= 4
                print("laplacian weight", loss_weights["laplacian"])
                print("normal consistency weight", loss_weights["normal"])
                print("lr vertices", training_config.lr_vertices)

                displacements.register_parameter(
                    "vertex_offsets",
                    torch.nn.Parameter(
                        torch.zeros(flame_canonical_mesh.vertices.shape), requires_grad=True
                    ),
                )
                displacements.canonical_vertices = flame_canonical_mesh.vertices
                displacements.vertices_shape = flame_canonical_mesh.vertices.shape
                displacements.to(device=device)
                optimizer_vertices = torch.optim.Adam(
                    list(displacements.parameters()), lr=training_config.lr_vertices
                )

            v_off = displacements()
            canonical_offset_vertices = flame_canonical_mesh.vertices + v_off
            mesh = flame_canonical_mesh.with_vertices(canonical_offset_vertices)

            shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)

            batched_verts = mesh.vertices.unsqueeze(0).repeat(training_config.batch_size, 1, 1)
            _, pose_features, transformations = FLAMEServer(
                expression_params=views_subset["flame_expression"],
                full_pose=views_subset["flame_pose"],
            )
            if training_config.ghostbone:
                transformations = torch.cat(
                    [
                        torch.eye(4)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(training_config.batch_size, -1, -1, -1)
                        .float()
                        .to(device),
                        transformations,
                    ],
                    1,
                )
            deformed_vertices = FLAMEServer.forward_pts_batch(
                pnts_c=batched_verts,
                betas=views_subset["flame_expression"],
                transformations=transformations,
                pose_feature=pose_features,
                shapedirs=shapedirs,
                posedirs=posedirs,
                lbs_weights=lbs_weights,
                dtype=torch.float32,
                map2_flame_original=True,
            )
            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
            # ==============================================================================================
            # R A S T E R I Z A T I O N
            # ==============================================================================================
            gbuffers = renderer.render_batch(
                views_subset["camera"],
                deformed_vertices.contiguous(),
                d_normals,
                channels=channels_gbuffer,
                with_antialiasing=True,
                canonical_v=mesh.vertices,
                canonical_idx=mesh.indices,
            )
            # ==============================================================================================
            # loss function 
            # ==============================================================================================
            ## ============== geometry regularization ==============================
            losses["normal"] = normal_consistency_loss(mesh)
            losses["laplacian"] = laplacian_loss(mesh)
            ## ============== color + regularization for color ==============================
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(
                gbuffers, views_subset, mesh, training_config.finetune_color, lgt
            )
            losses["shading"], pred_color, tonemapped_colors = shading_loss_batch(
                pred_color_masked, views_subset, training_config.batch_size
            )
            losses["perceptual_loss"] = VGGloss(
                tonemapped_colors[0], tonemapped_colors[1], iteration
            )
            losses["mask"] = mask_loss(views_subset["mask"], gbuffer_mask)
             ## ======= regularization color ========
            losses["albedo_regularization"] = albedo_regularization(
                _adaptive, shader, mesh, device, displacements, iteration
            )
            losses["white_light_regularization"] = white_light(cbuffers)
            losses["roughness_regularization"] = roughness_regularization(
                cbuffers["roughness"],
                views_subset["skin_mask"],
                views_subset["mask"],
                r_mean=training_config.r_mean,
            )
            losses["fresnel_coeff"] = spec_intensity_regularization(
                cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"]
            )
            ## ============== flame regularization ==============================
            if loss_weights["flame_regularization"] > 0:
                losses["flame_regularization"], gt_nn = flame_regularization(
                    FLAMEServer,
                    lbs_weights,
                    shapedirs,
                    posedirs,
                    mesh.vertices,
                    training_config.ghostbone,
                    iteration,
                    training_config.flame_mask,
                    views_subset=views_subset,
                    gbuffer=gbuffers,
                    weight_lbs=weight_flame_regularization,
                )

                if iteration in training_config.decay_flame:
                    print("Decaying flame regularization")
                    loss_weights["flame_regularization"] *= 0.5
            ## ============== diffusion regularization ==============================
            losses['diffusion_normal'] = diffusion_normal_regularization(gbuffers["normal"], views_subset["diffusion_normal"], views_subset["skin_mask"], views_subset["mask"])
            losses['diffusion_albedo'] = diffusion_albedo_regularization(cbuffers["albedo"], views_subset["diffusion_albedo"], views_subset["skin_mask"], views_subset["mask"])
            losses['diffusion_roughness'] = diffusion_roughness_regularization(cbuffers["roughness"], views_subset["diffusion_roughness"], views_subset["skin_mask"], views_subset["mask"])
            losses['diffusion_irradiance'] = diffusion_irradiance_regularization(cbuffers["irradiance"], views_subset["diffusion_irradiance"], views_subset["skin_mask"], views_subset["mask"])
            # ==============================================================================================
            # Aggregate losses
            # ==============================================================================================
            loss = torch.tensor(0.0, device=device)
            for k, v in losses.items():
                loss += v * loss_weights[k]
            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            optimizer_shader.zero_grad()
            optimizer_vertices.zero_grad()
            if training_config.train_deformer:
                optimizer_deformer.zero_grad()

            loss.backward()
            torch.cuda.synchronize()
            ### increase the gradients of positional encoding following tinycudnn
            if training_config.grad_scale and training_config.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0
            optimizer_shader.step()
            optimizer_vertices.step()
            if training_config.train_deformer:
                optimizer_deformer.step()
            progress_bar.set_postfix({"loss": loss.detach().cpu().item()})
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
                
            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (training_config.visualization_frequency > 0) and (iteration == 1 or iteration % training_config.visualization_frequency == 0):
                with torch.no_grad():
                    debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(training_config, mesh, debug_views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt)
                    ## ============== visualize ==============================
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, path_config.images_save_path(stage), iteration)
                    del debug_gbuffer, debug_cbuffers
            ## ============== save intermediate ==============================
            if (training_config.save_frequency > 0) and (iteration == 1 or iteration % training_config.save_frequency == 0):
                with torch.no_grad():
                    write_mesh(path_config.meshes_save_path(stage) / f"mesh_{iteration:06d}.obj", mesh.detach().to('cpu'))                                
                    shader.save(path_config.shaders_save_path(stage) / f'shader_{iteration:06d}.pt')
                    displacements.save(path_config.shaders_save_path(stage) / f'displacement_{iteration:06d}.pt')
                    deformer_net.save(path_config.shaders_save_path(stage) / f'deformer_{iteration:06d}.pt')


    end = time.time()
    total_time = (end - start) % 3600
    print("TIME TAKEN (mins):", int(total_time // 60))

    write_mesh(path_config.meshes_save_path(stage) / f"mesh_latest.obj", mesh.detach().to("cpu"))
    shader.save(path_config.shaders_save_path(stage) / f"shader_latest.pt")
    displacements.save(path_config.shaders_save_path(stage) / f"displacement_latest.pt")
    deformer_net.save(path_config.shaders_save_path(stage) / f"deformer_latest.pt")


def material_aware_training(
    run_name: str,
    working_dir: str,
    output_dir: str,
    flame_path: str,
    train_dir: list[str],
    eval_dir: list[str],
    input_dir: str,
    diffusion_dir: str,
    batch_size: int = 2,
    ghostbone: bool = True,
):
    path_config = PathConfig(
        working_dir, run_name, input_dir, train_dir, output_dir, flame_path, diffusion_dir
    )
    # stage 1
    default_stage_1_config = MaterialAwareTrainingConfig.default_stage_1_config(
        batch_size=batch_size, ghostbone=ghostbone
    )

    print("loading train views...")

    dataset_train = DatasetLoader(path_config, train_dir=train_dir, sample_ratio=1, pre_load=True)
    dataset_val = DatasetLoader(path_config, train_dir=eval_dir, sample_ratio=24, pre_load=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        collate_fn=dataset_train.collate,
        shuffle=True,
        drop_last=True,
    )
    view_indices = np.array(default_stage_1_config.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)
    del dataset_val
    FLAMEServer = build_flame_server(dataset_train, flame_path, ghostbone)

    write_config_to_json(
        path_config=path_config,
        train_config=default_stage_1_config,
        file_path=Path(f"{working_dir}/{output_dir}/{run_name}/config_stage1.json")
    )

    _material_aware_training(
        path_config,
        default_stage_1_config,
        dataset_train,
        dataloader_train,
        FLAMEServer,
        stage="stage_1",
        debug_views=debug_views
    )

    # stage 2
    default_stage_2_config = MaterialAwareTrainingConfig.default_stage_2_config(
        batch_size=batch_size, ghostbone=ghostbone
    )

    write_config_to_json(
        path_config=path_config,
        train_config=default_stage_2_config,
        file_path=Path(f"{working_dir}/{output_dir}/{run_name}/config_stage2.json")
    )

    _material_aware_training(
        path_config,
        default_stage_2_config,
        dataset_train,
        dataloader_train,
        FLAMEServer,
        stage="stage_2",
        debug_views=debug_views
    )


if __name__ == "__main__":
    material_aware_training(
        run_name="002",
        working_dir="/home/jsickert/adl4cv/",
        input_dir="DATA/001",
        train_dir=["MVI_1814", "MVI_1810"],
        output_dir="out",
        batch_size=2,
        flame_path="/home/jsickert/adl4cv/adl4cv_ws25-26_Relightable-Avatars/flare/flame/FLAME2020/generic_model.pkl",
        diffusion_dir=None,
    )
