import time
import os
import numpy as np
from pathlib import Path
from gpytoolbox import remesh_botsch
import torch
from types import SimpleNamespace
import sys
from functools import reduce
from itertools import product
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
import shutil
import random

project_root = Path().resolve().parent
sys.path.append(str(project_root))

from material_aware_flare.material_diffusion_regularization import (
    diffusion_albedo_regularization, 
    diffusion_irradiance_regularization, 
    diffusion_normal_regularization, 
    diffusion_roughness_regularization
)


sys.path.append("../flare") 
from flame.FLAME import FLAME
from flare.core import (
    Mesh, Renderer
)
from flare.losses import *
from flare.modules import (
    NeuralShader, get_deformer_network, Displacement
)
from flare.utils import (
    AABB, read_mesh, write_mesh,
    visualize_training,
    make_dirs, 
    set_defaults_finetune,
    save_individual_img
)
import nvdiffrec.render.light as light
from flare.dataset import DatasetLoader, dataset_util
from flare.dataset import *
from flare.metrics import metrics
import nvdiffrec.render.light as light


def format_output_dir(base: str, search_point: Tuple[float, float, float, float]):
    return f'{base}_{str(search_point[0]).replace("0.", "")}_{str(search_point[1]).replace("0.", "")}_{str(search_point[2]).replace("0.", "")}_{str(search_point[3]).replace("0.", "")}'


def _to_numpy(x):
    """Convert list/torch/tensor-like to numpy 1D array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x).flatten()


def ema(signal, alpha):
    """Exponential moving average smoothing. alpha in (0,1], higher = more smoothing."""
    if alpha <= 0:
        return signal
    out = np.empty_like(signal, dtype=float)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i-1]
    return out


def plot_losses(
    losses_history,
    keys=None,
    figsize=(10, 6),
    dpi=150,
    smoothing_alpha=0.0,   # 0 -> no smoothing, 0.1..0.3 typical
    show_raw=False,        # If True, plot raw + smoothed (when smoothing_alpha>0)
    log_scale=False,
    title=None,
    xlabel='Iteration',
    ylabel='Loss',
    ylim=None,
    legend_loc='best',
    linewidth=2.0,
    markers=None,
    save_path=None,
    fontsize=12
):
    if not isinstance(losses_history, dict):
        raise ValueError("losses_history must be a dict of lists/arrays.")

    # Default keys to all in the dict if not specified
    if keys is None:
        keys = list(losses_history.keys())

    # Setup figure style
    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "font.size": fontsize,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.2"
    })
    fig, ax = plt.subplots()

    # Color/marker cycles
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    n_colors = len(color_cycle) if color_cycle is not None else None
    if markers is None:
        markers = [None] * len(keys)

    plotted_any = False
    for i, key in enumerate(keys):
        if key not in losses_history:
            warnings.warn(f"Key '{key}' not found in losses_history — skipping.")
            continue

        y = _to_numpy(losses_history[key])
        if y.size == 0:
            warnings.warn(f"Key '{key}' contains no data — skipping.")
            continue

        x = np.arange(len(y))
        # Choose color/marker
        color = color_cycle[i % n_colors] if (color_cycle and n_colors) else None
        marker = markers[i] if i < len(markers) else None

        if smoothing_alpha > 0:
            y_s = ema(y, smoothing_alpha)
            if show_raw:
                ax.plot(x, y, linestyle='--', linewidth=max(0.8, linewidth*0.8),
                        alpha=0.6, label=f"{key} (raw)", color=color, marker=marker, markersize=4)
            ax.plot(x, y_s, linewidth=linewidth, label=f"{key} (smoothed, α={smoothing_alpha})",
                    color=color)
            last_val = y_s[-1]
        else:
            ax.plot(x, y, linewidth=linewidth, label=key, color=color, marker=marker, markersize=4)
            last_val = y[-1]

        # Add small text annotation at the end of the curve (avoids overlap by small x-offset)
        try:
            ax.annotate(f"{last_val:.4g}", xy=(x[-1], last_val),
                        xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=color)
        except Exception:
            # in case annotate fails for strange reasons, ignore
            pass

        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No valid losses were plotted. Check your keys / losses_history content.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2)

    if log_scale:
        ax.set_yscale('log')
        # better formatting for log scale
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.6)
    ax.minorticks_on()

    # Tight legend with a small title showing final values summary
    ax.legend(loc=legend_loc, fontsize=max(8, fontsize-1), framealpha=0.95)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved figure to: {save_path}")


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


def quantitative_eval(args, mesh, dataloader_validate, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):

    for it, views_subset in enumerate(dataloader_validate):
        with torch.no_grad():
            rgb_pred, gbuffer, cbuffer = run(args, mesh, views_subset, FLAMEServer, deformer_net, shader, renderer, device, 
                    channels_gbuffer, lgt=lgt)

        rgb_pred = rgb_pred * gbuffer["mask"]
        if save_each:
            save_individual_img(rgb_pred, views_subset, gbuffer["normal"], gbuffer["mask"], cbuffer, images_eval_save_path, iteration=None)
    ## ============== metrics ==============================
    gt_dir = Path(args.input_dir)
    if gt_dir is not None:
        eval_list = metrics.run(images_eval_save_path, gt_dir, args.eval_dir)

    with open(str(experiment_dir / "final_eval.txt"), 'a') as f:
        f.writelines("\n"+"w/o cloth result:"+"\n")
        f.writelines("\n"+"MAE | LPIPS | SSIM | PSNR | L1"+"\n")
        if gt_dir is not None:
            eval_list = [str(e) for e in eval_list]
            f.writelines(" ".join(eval_list))


def __loop__(config):
    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config.device >= 0:
        device = torch.device(f'cuda:{config.device}')
    dataset_train = DatasetLoader(config, train_dir=config.train_dir, sample_ratio=config.sample_idx_ratio, pre_load=True)
    dataset_val = DatasetLoader(config, train_dir=config.eval_dir, sample_ratio=24, pre_load=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, collate_fn=dataset_train.collate, shuffle=True, drop_last=True)
    view_indices = np.array(config.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)
    del dataset_val
    ### ============== load FLAME mesh ==============================
    flame_path = "/home/hleonhard/adl4cv_ws25-26_Relightable-Avatars/flare/flame/FLAME2020/generic_model.pkl"
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)
    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if config.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)
    for stage in [1, 2]:
        run_name = config.run_name if config.run_name is not None else config.input_dir.parent.name
        images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(config, run_name, config.finetune_color)
        ## ============== load mesh/train mesh ==============================
        if config.finetune_color:
            mesh_path = experiment_dir / "stage_1" / "meshes" / f"mesh_latest.obj"
            print("loading mesh from:", mesh_path)
            flame_canonical_mesh = read_mesh(mesh_path, device=device)
            flame_canonical_mesh.compute_connectivity()
            flame_canonical_mesh.to(device)
        else:
            if config.downsample:
                v_down, f_down = remesh_botsch(FLAMEServer.canonical_verts.squeeze(0).cpu().detach().numpy().astype(np.float64), 
                                                                        FLAMEServer.faces_tensor.cpu().numpy().astype(np.int32), h=float(config.downsample_ratio))
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
        renderer_visualization = Renderer(device=device)
        renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
        # ==============================================================================================
        # vertices
        # ==============================================================================================
        lr_vertices = config.lr_vertices
        displacements = Displacement(vertices_shape=flame_canonical_mesh.vertices.shape)
        displacements.to(device=device)
        optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=lr_vertices)
        # ==============================================================================================
        # deformation 
        # ==============================================================================================
        if config.train_deformer:
            model_path = None
            print("Training Deformer")
        else:
            print("Loading deformer network trained in the previous stage")
            config.weight_flame_regularization = 0.0
            model_path = Path(experiment_dir / "stage_1" / "network_weights" / f"deformer_latest.pt")
            assert os.path.exists(model_path)
        deformer_net = get_deformer_network(FLAMEServer, model_path=model_path, train=config.train_deformer, d_in=3, dims=config.deform_dims, 
                                               weight_norm=True, multires=0, num_exp=50, aabb=flame_mesh_aabb, ghostbone=config.ghostbone, device=device)
        if config.train_deformer:
            optimizer_deformer = torch.optim.Adam(list(deformer_net.parameters()), lr=config.lr_deformer)
        # ==============================================================================================
        # shading
        # ==============================================================================================
        lgt = light.create_env_rnd()    
        disentangle_network_params = {
            "material_mlp_ch": config.material_mlp_ch,
            "light_mlp_ch": config.light_mlp_ch,
            "material_mlp_dims": config.material_mlp_dims,
            "light_mlp_dims": config.light_mlp_dims
        }
        # Create the optimizer for the neural shader
        shader = NeuralShader(fourier_features=config.fourier_features,
                              activation=config.activation,
                              last_activation=torch.nn.Sigmoid(), 
                              disentangle_network_params=disentangle_network_params,
                              bsdf=config.bsdf,
                              aabb=flame_mesh_aabb,
                              device=device)
        params = list(shader.parameters()) 
        if config.weight_albedo_regularization > 0:
            from robust_loss_pytorch.adaptive import AdaptiveLossFunction
            _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
            params += list(_adaptive.parameters()) ## need to train it
        optimizer_shader = torch.optim.Adam(params, lr=config.lr_shader)
        # ==============================================================================================
        # Loss Functions
        # ==============================================================================================
        # Initialize the loss weights and losses
        loss_weights = {
            "mask": config.weight_mask,
            "normal": config.weight_normal,
            "laplacian": config.weight_laplacian,
            "shading": config.weight_shading,
            "perceptual_loss": config.weight_perceptual_loss,
            "albedo_regularization": config.weight_albedo_regularization,
            "roughness_regularization": config.weight_roughness_regularization,
            "white_light_regularization": config.weight_white_lgt_regularization,
            "fresnel_coeff": config.weight_fresnel_coeff,
            "diffusion_normal": config.diffusion_normal,
            "diffusion_albedo": config.diffusion_albedo,
            "diffusion_roughness": config.diffusion_roughness,
            "diffusion_irradiance": config.diffusion_irradiance,
        }
        if config.train_deformer:
            loss_weights["flame_regularization"] = 1.0 # we use the weight directly in loss function
        else:
            loss_weights["flame_regularization"] = 0.0
        losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
        if loss_weights["perceptual_loss"] > 0.0:
            VGGloss = VGGPerceptualLoss().to(device) # is loaded from flare losses
        shader.train()
        if config.train_deformer:
            deformer_net.train()
        displacements.train()
        # ==============================================================================================
        # T R A I N I N G
        # ==============================================================================================
        epochs = (config.iterations // len(dataloader_train)) + 1
        iteration = 0
        # progress_bar = tqdm(range(epochs))
        start = time.time()
        losses_history = {
            'total': [],
            'albedo_reg': [],
            'normal_reg': [],
            'roughness_reg': [],
            'irradiance_reg': [],
        }
        for epoch in range(epochs):
            for iter_, views_subset in enumerate(dataloader_train):
                iteration += 1
                # progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')
                # ==============================================================================================
                # upsample + remesh + reduce lr + freeze if required
                # ==============================================================================================
                if iteration in config.upsample_iterations and not config.finetune_color:
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
                    if iteration == config.upsample_iterations[0]:
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
                batched_verts = mesh.vertices.unsqueeze(0).repeat(config.batch_size, 1, 1)
                _, pose_features, transformations = FLAMEServer(expression_params=views_subset["flame_expression"], full_pose=views_subset["flame_pose"])
                if config.ghostbone:
                    transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(config.batch_size, -1, -1, -1).float().to(device), transformations], 1)
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
                pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, config.finetune_color, lgt)
                losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, config.batch_size)
                losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
                losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)
                ## ======= regularization color ========
                losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, displacements, iteration)
                losses['white_light_regularization'] = white_light(cbuffers)
                losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=config.r_mean)
                losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
                ## ============== flame regularization ==============================
                if loss_weights['flame_regularization'] > 0:
                    losses['flame_regularization'], gt_nn = flame_regularization(FLAMEServer, lbs_weights, shapedirs, posedirs, mesh.vertices, config.ghostbone, 
                                                                          iteration, config.flame_mask, views_subset=views_subset, gbuffer=gbuffers, 
                                                                          weight_lbs=config.weight_flame_regularization)
                    if iteration in config.decay_flame:
                        print("Decaying flame regularization")
                        loss_weights['flame_regularization'] *= 0.5
                ## ============== diffusion regularization ==============================
                losses['diffusion_normal'] = diffusion_normal_regularization(gbuffers["normal"], views_subset["diffusion_normal"], views_subset["skin_mask"], views_subset["mask"])
                losses['diffusion_albedo'] = diffusion_albedo_regularization(cbuffers["albedo"], views_subset["diffusion_albedo"], views_subset["skin_mask"], views_subset["mask"])
                losses['diffusion_roughness'] = diffusion_roughness_regularization(cbuffers["roughness"], views_subset["diffusion_roughness"], views_subset["skin_mask"], views_subset["mask"])
                losses['diffusion_irradiance'] = diffusion_irradiance_regularization(cbuffers["irradiance"], views_subset["diffusion_irradiance"], views_subset["skin_mask"], views_subset["mask"])
                loss = torch.tensor(0., device=device) 
                for k, v in losses.items():
                    loss += v * loss_weights[k]
                # ==============================================================================================
                # Optimizer step
                # ==============================================================================================
                optimizer_shader.zero_grad()
                optimizer_vertices.zero_grad()
                if config.train_deformer:
                    optimizer_deformer.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
                ### increase the gradients of positional encoding following tinycudnn
                if config.grad_scale and config.fourier_features == "hashgrid":
                    shader.fourier_feature_transform.params.grad /= 8.0
                optimizer_shader.step()
                optimizer_vertices.step()
                if config.train_deformer:
                    optimizer_deformer.step()
                # progress_bar.set_postfix({'loss': loss.detach().cpu().item()})
                losses_history['total'].append(loss.detach().cpu().item())
                losses_history['albedo_reg'].append(losses['diffusion_albedo'].detach().cpu().item())
                losses_history['normal_reg'].append(losses['diffusion_normal'].detach().cpu().item())
                losses_history['roughness_reg'].append(losses['diffusion_roughness'].detach().cpu().item())
                losses_history['irradiance_reg'].append(losses['diffusion_irradiance'].detach().cpu().item())
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
                if (config.visualization_frequency > 0) and (iteration == 1 or iteration % config.visualization_frequency == 0):
                    with torch.no_grad():
                        debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(config, mesh, debug_views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt)
                        ## ============== visualize ==============================
                        visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration)
                        del debug_gbuffer, debug_cbuffers
                ## ============== save intermediate ==============================
                if (config.save_frequency > 0) and (iteration == 1 or iteration % config.save_frequency == 0):
                    with torch.no_grad():
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh.detach().to('cpu'))                                
                        shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')
                        displacements.save(shaders_save_path / f'displacement_{iteration:06d}.pt')
                        deformer_net.save(shaders_save_path / f'deformer_{iteration:06d}.pt')
        end = time.time()
        total_time = ((end - start) % 3600)
        print("TIME TAKEN (mins):", int(total_time // 60))
        # ==============================================================================================
        # s a v e
        # ==============================================================================================
        with open(experiment_dir / "args.txt", "w") as text_file:
            print(f"{config}", file=text_file)
        write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.detach().to('cpu'))
        shader.save(shaders_save_path / f'shader_latest.pt')
        displacements.save(shaders_save_path / f'displacement_latest.pt')
        deformer_net.save(shaders_save_path / f'deformer_latest.pt')
        # stage 2 training
        config.finetune_color = True
        config.final_iter = config.iterations
        config.lr_vertices = 1e-5
        config.train_deformer = False
        config.iterations = 1000
        config.sample_idx_ratio = 1
        config.fourier_features = "hashgrid"
        config.material_mlp_dims = [64, 64]
        config.light_mlp_dims = [64, 64]
    # save loss history
    plot_losses(
        losses_history,
        keys=['total', 'albedo_reg', 'normal_reg', 'roughness_reg' 'irradiance_reg'],
        smoothing_alpha=0.025,   # set to 0 for no smoothing
        show_raw=False,
        log_scale=False,
        title=f"Training Losses - DiffuionRenderer ({config.diffusion_normal}, {config.diffusion_albedo}, {config.diffusion_roughness}, {config.diffusion_irradiance})",
        save_path=f'{experiment_dir.__str__()}/training_losses.png'
    )
    if config.finetune_color:        
        ## ============== free memory before evaluation ==============================
        del dataset_train, dataloader_train, debug_views, views_subset
        print("E V A L U A T I O N")
        dataset_val = DatasetLoader(config, train_dir=config.eval_dir, sample_ratio=1, pre_load=True)
        dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=4, collate_fn=dataset_val.collate)
        quantitative_eval(config, mesh, dataloader_validate, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, experiment_dir
                        , images_eval_save_path / "qualitative_results", lgt=lgt, save_each=True)
    data = np.load(images_eval_save_path / "qualitative_results" / "results_no_cloth_rgb.npz")
    for item in data.files:
        if item in ["l1_l", "perceptual_l", "ssim_l", "psnr_l"]:
            print(f"Mean {item}: {np.mean(data[item])}")
        if item == 'filenames':
            print(f"Evalauted a total of {len(data[item])} points")


def main():
    search_space = {
        "diffusion_normal": [0.0, 0.01, 0.1],
        "diffusion_albedo": [0.0, 0.01, 0.1],
        "diffusion_roughness": [0.0, 0.01, 0.1],
        "diffusion_irradiance": [0.0, 0.01, 0.1],
    }
    num_search_configs = reduce(lambda x, y: x * len(y), search_space.values(), 1)
    search_points = list(product(*search_space.values()))
    search_points = search_points[1:]
    random.shuffle(search_points)
    for search_point in search_points:
        config = {
            'config': None,
            'run_name': 'rgbx_05_05_05_0',
            'batch_size': 2,
            # path
            'input_dir': Path("/home/hleonhard/data/flare_subject_data/001"),
            'train_dir': ["MVI_1814", "MVI_1810"],
            'eval_dir': ["MVI_1812"],
            'working_dir': Path("/home/hleonhard/data/flare_train_setup"),
            'output_dir': Path("out"),
            # misc
            'sample_idx_ratio': 1,
            'device': 0,
            'finetune_color': False,
            # iters
            'iterations': 2000, 
            'final_iter': 1500,
            'upsample_iterations': [500],
            'save_frequency': 300,
            'visualization_frequency': 100,
            'visualization_views': [15, 25, 27, 21, 26],
            # 'downsample' default is set via set_defaults
            'downsample': False,
            'downsample_ratio': 0.03,
            'grad_scale': False, # Default for action='store_true' is False unless set otherwise
            # flame
            'decay_flame': [100],
            'flame_mask': False,
            # lr
            'lr_vertices': 1e-3,
            'lr_shader': 1e-3,
            'lr_deformer': 1e-3,
            # loss weights
            'weight_mask': 2.0,
            'weight_normal': 0.1,
            'weight_laplacian': 60.0,
            'weight_shading': 1.0,
            'weight_perceptual_loss': 0.1,
            'weight_albedo_regularization': 0.01,
            'weight_flame_regularization': 10.0,
            'weight_white_lgt_regularization': 1.0,
            'weight_roughness_regularization': 0.1,
            'weight_fresnel_coeff': 0.01,
            # diffusion regularization
            'diffusion_dir': Path("/home/hleonhard/data/flare_diffusion_channels/rgbx/001"),
            "diffusion_normal": 0.05,
            "diffusion_albedo": 0.05,
            "diffusion_roughness": 0.05,
            "diffusion_irradiance": 0.0,
            'r_mean': 0.500,
            # neural shader
            'fourier_features': 'positional',
            'activation': 'relu',
            'bsdf': 'pbr_shading',
            'deform_d_out': 128,
            'light_mlp_ch': 3,
            'light_mlp_dims': [64, 64],
            'material_mlp_dims': [128, 128, 128, 128, 128],
            'material_mlp_ch': 5,
            # ghostbone/train_deformer (defaults are set via set_defaults)
            'ghostbone': True,
            'train_deformer': True,
            'deform_dims': [128, 128, 128, 128]
        }
        config = SimpleNamespace(**config)
        config.run_name = format_output_dir('rgbx', search_point)
        config.diffusion_normal = search_point[0]
        config.diffusion_albedo = search_point[1]
        config.diffusion_roughness = search_point[2]
        config.diffusion_irradiance = search_point[3]
        print("=="*50)
        print(config.run_name)
        print("=="*50)
        repeats = 0
        while True:
            try:
                __loop__(config)
                break # success
            except ValueError as e:
                depracted_path = Path("/home/hleonhard/data/flare_train_setup/out/") / config.run_name
                if depracted_path.exists():
                    shutil.rmtree(depracted_path) 
                print(f"Restarting training for {config} -> deleted {depracted_path}")
                repeats += 1
                if repeats == 3:
                    print(f"reached 3 repeats, stopping now")
                    break
    print(f"==" * 50)
    print(f"Finished grid search")
    print(f"==" * 50)

if __name__ == "__main__":
    # how to run
    # screen -S grid_search
    # conda activate flare
    # python grid_search.py > output_grid_search.log 2>&1
    # Ctrl+a d # to detach the session
    # screen -ls # view all running sessions
    # screen -r grid_search # detach session again
    # screen -S grid_search -X quit # kill the session
    main()