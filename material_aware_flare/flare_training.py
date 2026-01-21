from omegaconf import OmegaConf
import argparse
import os
import numpy as np
from pathlib import Path
from gpytoolbox import remesh_botsch
import torch
from tqdm.notebook import tqdm
# regularization related imports
from material_aware_flare.material_diffusion_regularization import (
    diffusion_albedo_regularization, 
    diffusion_irradiance_regularization, 
    diffusion_normal_regularization, 
    diffusion_roughness_regularization
)
# flare related imports
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
    # set_defaults_finetune
)
import nvdiffrec.render.light as light
from flare.dataset import DatasetLoader, dataset_util
from flare.dataset import *
from flare.metrics import metrics


def build_arg_parser():
    p = argparse.ArgumentParser()

    # top-level
    p.add_argument("--config")
    p.add_argument("--run_name")
    p.add_argument("--batch_size", type=int, default=2)

    # paths
    p.add_argument("--input_dir")
    p.add_argument("--train_dir", nargs="+")
    p.add_argument("--eval_dir", nargs="+")
    p.add_argument("--working_dir")
    p.add_argument("--output_dir")

    # misc
    p.add_argument("--sample_idx_ratio", type=float)
    p.add_argument("--device", type=int)

    # iterations
    p.add_argument("--iterations", type=int)
    p.add_argument("--final_iter", type=int)
    p.add_argument("--upsample_iterations", nargs="+", type=int)
    p.add_argument("--save_frequency", type=int)
    p.add_argument("--visualization_frequency", type=int)
    p.add_argument("--visualization_views", nargs="+", type=int)
    p.add_argument("--downsample_ratio", type=float)

    # flame
    p.add_argument("--decay_flame", nargs="+", type=int)
    p.add_argument('--flame_path', type=str)

    # lr
    p.add_argument("--lr_vertices", type=float)
    p.add_argument("--lr_shader", type=float)
    p.add_argument("--lr_deformer", type=float)

    # loss weights
    p.add_argument("--weight_mask", type=float)
    p.add_argument("--weight_normal", type=float)
    p.add_argument("--weight_laplacian", type=float)
    p.add_argument("--weight_shading", type=float)
    p.add_argument("--weight_perceptual_loss", type=float)
    p.add_argument("--weight_albedo_regularization", type=float)
    p.add_argument("--weight_flame_regularization", type=float)
    p.add_argument("--weight_white_lgt_regularization", type=float)
    p.add_argument("--weight_roughness_regularization", type=float)
    p.add_argument("--weight_fresnel_coeff", type=float)

    # diffusion
    p.add_argument("--diffusion_dir")
    p.add_argument("--diffusion_normal", type=float)
    p.add_argument("--diffusion_albedo", type=float)
    p.add_argument("--diffusion_roughness", type=float)
    p.add_argument("--diffusion_irradiance", type=float)
    p.add_argument("--r_mean", type=float)

    # neural shader
    p.add_argument("--fourier_features")
    p.add_argument("--activation")
    p.add_argument("--bsdf")
    p.add_argument("--deform_d_out", type=int)
    p.add_argument("--light_mlp_ch", type=int)
    p.add_argument("--light_mlp_dims", nargs="+", type=int)
    p.add_argument("--material_mlp_dims", nargs="+", type=int)
    p.add_argument("--material_mlp_ch", type=int)

    # ghostbone / deformer
    p.add_argument("--deform_dims", nargs="+", type=int)

    return p

def build_datasets(config):
    dataset_train = DatasetLoader(config, train_dir=config.train_dir, sample_ratio=config.sample_idx_ratio, pre_load=True)
    dataset_val = DatasetLoader(config, train_dir=config.eval_dir, sample_ratio=24, pre_load=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, collate_fn=dataset_train.collate, shuffle=True, drop_last=True)
    view_indices = np.array(config.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)
    del dataset_val
    return dataset_train, dataloader_train, debug_views

def build_flame(config, device: torch.device, dataset_train: DatasetLoader):
    
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(config.flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)
    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if config.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)
    return FLAMEServer

def build_mesh(flame_server: FLAME, config, device: torch.device, meshes_save_path: str, experiment_dir: str):
    ## ============== load mesh/train mesh ==============================
    if config.finetune_color:
        mesh_path = experiment_dir / "stage_1" / "meshes" / f"mesh_latest.obj"
        print("loading mesh from:", mesh_path)
        flame_canonical_mesh = read_mesh(mesh_path, device=device)
        flame_canonical_mesh.compute_connectivity()
        flame_canonical_mesh.to(device)
    else:
        if config.downsample:
            v_down, f_down = remesh_botsch(flame_server.canonical_verts.squeeze(0).cpu().detach().numpy().astype(np.float64), 
                                                                    flame_server.faces_tensor.cpu().numpy().astype(np.int32), h=float(config.downsample_ratio))
            verts = np.ascontiguousarray(v_down)
            faces = np.ascontiguousarray(f_down)
            print("Downsampled:", verts.shape, faces.shape)
        else:
            verts = flame_server.canonical_verts.squeeze(0)
            faces = flame_server.faces_tensor
        flame_canonical_mesh: Mesh = None
        flame_canonical_mesh = Mesh(verts, faces, device=device)
        flame_canonical_mesh.compute_connectivity()
        write_mesh(Path(meshes_save_path / "init_mesh.obj"), flame_canonical_mesh.to('cpu'))
        return flame_canonical_mesh
    
def init_losses(config, device: torch.device):
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
    print(loss_weights)
    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device) # is loaded from flare losses
    return loss_weights, losses, VGGloss

def run(args, mesh, views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt):
    """
    util func for evaluation
    """
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

def make_paths(config, fields):
    from pathlib import Path
    for f in fields:
        if getattr(config, f, None) is not None:
            setattr(config, f, Path(getattr(config, f)))
    return config


def flare_pipeline(config_str: str="config/default_config.yaml"):
    parser = build_arg_parser()
    args = parser.parse_args()
    # Load config file
    cfg_file = OmegaConf.load(config_str)
    cli_dict = {k: v for k, v in vars(args).items() if v is not None}
    cfg_cli = OmegaConf.create(cli_dict)   
    # Merge config file + CLI (CLI takes precedence)
    config = OmegaConf.merge(cfg_file, cfg_cli)
    path_fields = ["input_dir", "working_dir", "output_dir", "diffusion_dir"]
    config = make_paths(config, path_fields)
    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config.device >= 0:
        device = torch.device(f'cuda:{config.device}')
    # setup dataset
    dataset_train, dataloader_train, debug_views = build_datasets(config=config)
    ### ============== load FLAME mesh ==============================
    FLAMEServer = build_flame(config=config, device=device, dataset_train=dataset_train)
    ### ============== conig params ==============================
    run_name = config.run_name if config.run_name is not None else config.input_dir.parent.name
    images_save_path, _, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(config, run_name, config.finetune_color)
    # write config to experiment dir
    with open(f"{experiment_dir}/config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp)
    # print(f'Setup dirs \n{images_save_path}\n{images_eval_save_path}\n{meshes_save_path}\n{shaders_save_path}\n{experiment_dir}')
    ## ============== load mesh/train mesh ==============================
    flame_canonical_mesh = build_mesh(flame_server=FLAMEServer, config=config, device=device, meshes_save_path=meshes_save_path, experiment_dir=experiment_dir)
    ## ============== renderer ==============================
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [torch.min(flame_canonical_mesh.vertices, dim=0).values, torch.max(flame_canonical_mesh.vertices, dim=0).values]
    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    ## ============== vertices ==============================
    lr_vertices = config.lr_vertices
    displacements = Displacement(vertices_shape=flame_canonical_mesh.vertices.shape)
    displacements.to(device=device)
    optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=config.lr_vertices)
    ## ============== deformation ==============================
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
     ## ============== shading ==============================
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
    # ============== losses ==============================
    loss_weights, losses, VGGloss = init_losses(config=config, device=device)
    shader.train()
    if config.train_deformer:
        deformer_net.train()
    displacements.train()

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================
    epochs = (config.iterations // len(dataloader_train)) + 1
    iteration = 0
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        for _, views_subset in enumerate(dataloader_train):
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')

            # ==============================================================================================
            # upsample + remesh + reduce lr + freeze if required
            # ==============================================================================================
            if iteration in config.upsample_iterations and not config.finetune_color:
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
    print(f"Finised training, checkout {experiment_dir}")


if __name__ == "__main__":
    # provide top level rop 
    # export PYTHONPATH=/home/hleonhard/adl4cv_ws25-26_Relightable-Avatars/flare:$PYTHONPATH
    flare_pipeline()