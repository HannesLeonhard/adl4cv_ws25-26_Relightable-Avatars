from flame.FLAME import FLAME
from flare.core import Mesh, Renderer
from flare.losses import *
from flare.modules import NeuralShader, get_deformer_network, Displacement
from flare.utils import (
    AABB,
    read_mesh,
    write_mesh,
    make_dirs,
)
import nvdiffrec.render.light as light
from flare.dataset import DatasetLoader
from flare.dataset import *
import nvdiffrec.render.light as light
import torch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from gpytoolbox import remesh_botsch
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
import time
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_loop(
    working_dir: str,
    input_dir: str,
    output_dir: str,
    train_dir: list[str],
    run_name: str,
    flame_path,
    finetune_color: bool,
    train_deformer: bool,
    batch_size: int = 2,
    iterations: int = 1500,
    downsample: bool = False,
    downsample_ratio: float = 0.03,
    lr_vertices: float = 1e-3,
    lr_deformer: float = 1e-3,
    lr_shader: float = 1e-3,
    material_mlp_ch: int = 5,
    light_mlp_ch: int = 3,
    material_mlp_dims: list[int] = [128, 128, 128, 128, 128],
    light_mlp_dims: list[int] = [64, 64],
    weight_mask: float = 2.0,
    weight_normal: float = 0.1,
    weight_laplacian: float = 60.0,
    weight_shading: float = 1.0,
    weight_perceptual_loss: float = 0.1,
    weight_flame_regularization: float = 10.0,
    weight_albedo_regularization: float = 0.01,
    weight_roughness_regularization: float = 0.1,
    weight_white_lgt_regularization: float = 1.0,
    weight_fresnel_coeff: float = 0.01,
    fourier_features: str = "positional",
    activation: str = "relu",
    bsdf: str = "pbr_shading",
    deform_dims: list[int] = [128, 128, 128, 128],
    r_mean: float = 0.500,
    flame_mask: bool = False,
    decay_flame: list[int] = [100],
    grad_scale: bool = False,
    upsample_iterations: list[int] = [500],
    ghostbone: bool = True,
):
    _config = {
        "working_dir": Path(working_dir),
        "input_dir": Path(input_dir),
        "output_dir": Path(output_dir),
    }

    config = SimpleNamespace(**_config)

    dataset_train = DatasetLoader(config, train_dir=train_dir, sample_ratio=1, pre_load=True)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        collate_fn=dataset_train.collate,
        shuffle=True,
        drop_last=True,
    )

    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)
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

    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = (
        make_dirs(config, run_name, finetune_color)
    )

    # load mesh
    if finetune_color:
        mesh_path = Path(experiment_dir) / "stage_1" / "meshes" / f"mesh_latest.obj"
        flame_canonical_mesh = read_mesh(mesh_path, device=device)
        flame_canonical_mesh.compute_connectivity()
        flame_canonical_mesh.to(device)
        print("loading mesh from:", mesh_path)
    else:
        if downsample:
            v_down, f_down = remesh_botsch(
                FLAMEServer.canonical_verts.squeeze(0).cpu().detach().numpy().astype(np.float64),
                FLAMEServer.faces_tensor.cpu().numpy().astype(np.int32),
                h=float(downsample_ratio),
            )
            verts = np.ascontiguousarray(v_down)
            faces = np.ascontiguousarray(f_down)
            print("Downsampled:", verts.shape, faces.shape)
        else:
            verts = FLAMEServer.canonical_verts.squeeze(0)
            faces = FLAMEServer.faces_tensor

        flame_canonical_mesh: Mesh = None
        flame_canonical_mesh = Mesh(verts, faces, device=device)
        flame_canonical_mesh.compute_connectivity()
        write_mesh(Path(meshes_save_path / "init_mesh.obj"), flame_canonical_mesh.to("cpu"))

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

    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(
        dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5
    )

    # vertices
    displacements = Displacement(vertices_shape=flame_canonical_mesh.vertices.shape)
    displacements.to(device=device)
    optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=lr_vertices)

    # deformation
    if train_deformer:
        model_path = None
        print("Training deformer")
    else:
        print("Loading deformer network trained in the previous stage")
        weight_flame_regularization = 0.0
        model_path = Path(experiment_dir / "stage_1" / "network_weights" / f"deformer_latest.pt")

    deformer_net = get_deformer_network(
        FLAMEServer,
        model_path=model_path,
        train=train_deformer,
        d_in=3,
        dims=deform_dims,
        weight_norm=True,
        multires=0,
        num_exp=50,
        aabb=flame_mesh_aabb,
        ghostbone=ghostbone,
        device=device,
    )

    if train_deformer:
        optimizer_deformer = torch.optim.Adam(list(deformer_net.parameters()), lr=lr_deformer)

    # shader
    lgt = light.create_env_rnd()
    disentangle_network_params = {
        "material_mlp_ch": material_mlp_ch,
        "light_mlp_ch": light_mlp_ch,
        "material_mlp_dims": material_mlp_dims,
        "light_mlp_dims": light_mlp_dims,
    }
    shader = NeuralShader(
        fourier_features=fourier_features,
        activation=activation,
        last_activation=torch.nn.Sigmoid(),
        disentangle_network_params=disentangle_network_params,
        bsdf=bsdf,
        aabb=flame_mesh_aabb,
        device=device,
    )
    params = list(shader.parameters())
    if weight_albedo_regularization > 0:
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters())  ## need to train it
    optimizer_shader = torch.optim.Adam(params, lr=lr_shader)

    # loss functions
    loss_weights = {
        "mask": weight_mask,
        "normal": weight_normal,
        "laplacian": weight_laplacian,
        "shading": weight_shading,
        "perceptual_loss": weight_perceptual_loss,
        "albedo_regularization": weight_albedo_regularization,
        "roughness_regularization": weight_roughness_regularization,
        "white_light_regularization": weight_white_lgt_regularization,
        "fresnel_coeff": weight_fresnel_coeff,
    }

    if train_deformer:
        loss_weights["flame_regularization"] = 1.0
    else:
        loss_weights["flame_regularization"] = 0.0

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)

    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    shader.train()
    if train_deformer:
        deformer_net.train()
    displacements.train()

    epochs = (iterations // len(dataloader_train)) + 1
    iteration = 0
    start = time.time()
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        for iter, views in enumerate(dataloader_train):
            iteration += 1
            progress_bar.set_description(desc=f"Epoch {epoch}, Iter {iteration}")

            if iteration in upsample_iterations and not finetune_color:
                print("Upsampling iteration: ", iteration)
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

                if iteration == upsample_iterations[0]:
                    lr_vertices *= 0.75
                    loss_weights["laplacian"] *= 4
                    loss_weights["normal"] *= 4

                print("laplacian weight", loss_weights["laplacian"])
                print("normal consistency weight", loss_weights["normal"])
                print("lr vertices", lr_vertices)

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
                    list(displacements.parameters()), lr=lr_vertices
                )

            v_off = displacements()
            canonical_offset_vertices = flame_canonical_mesh.vertices + v_off
            mesh = flame_canonical_mesh.with_vertices(canonical_offset_vertices)

            shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
            batched_verts = mesh.vertices.unsqueeze(0).repeat(batch_size, 1, 1)
            _, pose_features, transformations = FLAMEServer(
                expression_params=views["flame_expression"], full_pose=views["flame_pose"]
            )
            if ghostbone:
                transformations = torch.cat(
                    [
                        torch.eye(4)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(batch_size, -1, -1, -1)
                        .float()
                        .to(device),
                        transformations,
                    ],
                    1,
                )
            deformed_vertices = FLAMEServer.forward_pts_batch(
                pnts_c=batched_verts,
                betas=views["flame_expression"],
                transformations=transformations,
                pose_feature=pose_features,
                shapedirs=shapedirs,
                posedirs=posedirs,
                lbs_weights=lbs_weights,
                dtype=torch.float32,
                map2_flame_original=True,
            )
            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

            # rasterization
            gbuffers = renderer.render_batch(
                views["camera"],
                deformed_vertices.contiguous(),
                d_normals,
                channels=channels_gbuffer,
                with_antialiasing=True,
                canonical_v=mesh.vertices,
                canonical_idx=mesh.indices,
            )

            losses["normal"] = normal_consistency_loss(mesh)
            losses["laplacian"] = laplacian_loss(mesh)

            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(
                gbuffers, views, mesh, finetune_color, lgt
            )
            losses["shading"], pred_color, tonemapped_colors = shading_loss_batch(
                pred_color_masked, views, batch_size
            )
            losses["perceptual_loss"] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            losses["mask"] = mask_loss(views["mask"], gbuffer_mask)

            losses["albedo_regularization"] = albedo_regularization(
                _adaptive, shader, mesh, device, displacements, iteration
            )
            losses["white_light_regularization"] = white_light(cbuffers)
            losses["roughness_regularization"] = roughness_regularization(
                cbuffers["roughness"], views["skin_mask"], views["mask"], r_mean=r_mean
            )
            losses["fresnel_coeff"] = spec_intensity_regularization(
                cbuffers["ko"], views["skin_mask"], views["mask"]
            )

            if loss_weights["flame_regularization"] > 0:
                losses["flame_regularization"], gt_nn = flame_regularization(
                    FLAMEServer,
                    lbs_weights,
                    shapedirs,
                    posedirs,
                    mesh.vertices,
                    ghostbone,
                    iteration,
                    flame_mask,
                    views_subset=views,
                    gbuffer=gbuffers,
                    weight_lbs=weight_flame_regularization,
                )

                if iteration in decay_flame:
                    print("Decaying flame regularization")
                    loss_weights["flame_regularization"] *= 0.5

            loss = torch.tensor(0.0, device=device)
            for k, v in losses.items():
                loss += v * loss_weights[k]

            optimizer_shader.zero_grad()
            optimizer_vertices.zero_grad()
            if train_deformer:
                optimizer_deformer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()

            if grad_scale and fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0
            optimizer_shader.step()
            optimizer_vertices.step()
            if train_deformer:
                optimizer_deformer.step()
            progress_bar.set_postfix({"loss": loss.detach().cpu().item()})

    end = time.time()
    total_time = (end - start) % 3600
    print("TIME TAKEN (mins):", int(total_time // 60))

    print(shaders_save_path.exists())

    write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.detach().to("cpu"))
    shader.save(shaders_save_path / f"shader_latest.pt")
    displacements.save(shaders_save_path / f"displacement_latest.pt")
    deformer_net.save(shaders_save_path / f"deformer_latest.pt")

    del dataloader_train


if __name__ == "__main__":
    # train_loop(
    #     working_dir="/home/jsickert/adl4cv/",
    #     output_dir="out",
    #     input_dir="DATA/001",
    #     train_dir=["MVI_1814", "MVI_1810"],
    #     run_name="001",
    #     finetune_color=False,
    #     train_deformer=True,
    #     flame_path="/home/jsickert/adl4cv/adl4cv_ws25-26_Relightable-Avatars/flare/flame/FLAME2020/generic_model.pkl",
    # )

    train_loop(
        working_dir="/home/jsickert/adl4cv/",
        output_dir="out",
        input_dir="DATA/001",
        train_dir=["MVI_1814", "MVI_1810"],
        run_name="001",
        finetune_color=True,
        train_deformer=False,
        lr_vertices=1e-5,
        iterations=1000,
        fourier_features="hashgrid",
        material_mlp_dims=[64, 64],
        light_mlp_dims=[64, 64],
        flame_path="/home/jsickert/adl4cv/adl4cv_ws25-26_Relightable-Avatars/flare/flame/FLAME2020/generic_model.pkl",
    )
