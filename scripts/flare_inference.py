import types
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import nvdiffrec.render.light as light
import torch
from flame.FLAME import FLAME

from flare.core import Mesh, Renderer
from flare.dataset import DatasetLoader
from flare.modules import ForwardDeformer, NeuralShader, get_deformer_network
from flare.utils import AABB, read_mesh

device = torch.device("cuda:0")


def build_flare_pipeline(
    dataset_val: DatasetLoader,
    flame_path: str,
    mesh_path: str,
    deformer_path: str,
    shader_path: str,
    train_dir: list[str],
    ghostbone: bool = True,
):
    # flame
    flame_shape = dataset_val.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)
    flame_canonical_mesh = Mesh(FLAMEServer.v_template, FLAMEServer.faces_tensor, device=device)
    flame_canonical_mesh.compute_connectivity()
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [
        torch.min(flame_canonical_mesh.vertices, dim=0).values,
        torch.max(flame_canonical_mesh.vertices, dim=0).values,
    ]
    FLAMEServer.canonical_exp = dataset_val.get_mean_expression_train(train_dir).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    (
        FLAMEServer.canonical_verts,
        FLAMEServer.canonical_pose_feature,
        FLAMEServer.canonical_transformations,
    ) = FLAMEServer(
        expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose
    )
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)
    flame_canonical_mesh.vertices = FLAMEServer.canonical_verts.squeeze(0)

    # mesh
    mesh = read_mesh(mesh_path, device=device)
    mesh.compute_connectivity()
    mesh.to(device)

    # renderer
    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_val, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ["mask", "position", "normal", "canonical_position"]

    # deformer
    multires = 0
    deformer_net = get_deformer_network(
        FLAMEServer,
        model_path=deformer_path,
        train=False,
        d_in=3,
        dims=[128, 128, 128, 128],
        weight_norm=True,
        multires=multires,
        num_exp=50,
        aabb=aabb,
        ghostbone=ghostbone,
        device=device,
    )
    if ghostbone:
        FLAMEServer.canonical_transformations = torch.cat(
            [
                torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device),
                FLAMEServer.canonical_transformations,
            ],
            1,
        )

    # shader
    shader = NeuralShader.load(shader_path, device=device)
    lgt = light.create_env_rnd()

    shader.eval()
    deformer_net.eval()

    return mesh, FLAMEServer, deformer_net, shader, renderer, channels_gbuffer, lgt


@torch.no_grad()
def run_inference(
    views,
    mesh: Mesh,
    FLAMEServer: FLAME,
    deformer_net: ForwardDeformer,
    shader: NeuralShader,
    renderer: Renderer,
    channels_gbuffer: list[str],
    lgt: light.EnvironmentLight,
    ghostbone: bool = True,
    finetune_color: bool = True,
):
    shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
    eval_vertices = mesh.vertices
    batched_verts = eval_vertices.unsqueeze(0).repeat(views["img"].shape[0], 1, 1)

    _, pose_features, transformations = FLAMEServer(
        expression_params=views["flame_expression"], full_pose=views["flame_pose"]
    )
    if ghostbone:
        transformations = torch.cat(
            [
                torch.eye(4)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(views["img"].shape[0], -1, -1, -1)
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

    gbuffers = renderer.render_batch(
        views["camera"],
        deformed_vertices.contiguous(),
        d_normals,
        channels=channels_gbuffer,
        with_antialiasing=True,
        canonical_v=mesh.vertices,
        canonical_idx=mesh.indices,
    )

    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, finetune_color, lgt)

    return rgb_pred, gbuffers, cbuffers


def visualize_image_batch(images: torch.Tensor, save_path: Optional[str] = None):
    bz = images.shape[0]
    images_np = images.cpu().numpy()
    images_np = np.clip(images_np, 0.0, 1.0)

    fig, axes = plt.subplots(1, bz, figsize=(5 * bz, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if bz == 1:
        axes = [axes]

    for i in range(bz):
        ax = axes[i]
        ax.imshow(images_np[i], cmap="gray")
        ax.set_title(f"Image {i + 1}")
        ax.axis("off")

    if save_path:
        fig.savefig(save_path)

    plt.show()


def visualize_image_buffers(image_dict: dict[str, torch.Tensor], save_path: Optional[str] = None):
    keys = list(image_dict.keys())
    num_cols = len(keys)
    num_rows = image_dict[keys[0]].shape[0]

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), squeeze=False
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.1)

    for r in range(num_rows):
        for c, key in enumerate(keys):
            images = image_dict[key]
            images_np = images.cpu().numpy()
            images_np = np.clip(images_np, 0.0, 1.0)
            ax = axes[r, c]
            ax.imshow(images_np[r], cmap="gray")
            ax.axis("off")

            if r == 0:
                ax.set_title(key, fontsize=12)

    if save_path:
        fig.savefig(save_path)

    plt.show()


def prepare_mask_for_metrics(pred: torch.Tensor, views: dict[str, torch.Tensor]):
    masked_pred = pred.cpu()  # gbuffers["mask"] was applied before (0/1)
    masked_gt = views["img"].cpu()  # gt mask was applied before (0/1)
    mask = views["mask"].cpu()  # 0/1 mask from file
    semantics = views["skin_mask"].cpu()

    assert (masked_gt == masked_gt * mask).all()

    # consider only relevant areas, i.e neglect clothing
    # see flare/dataset/dataset_util.py:_load_semantics
    mask_cloth = semantics[..., 4:5]
    mask = mask * (1 - mask_cloth)

    # apply updated mask on gt and pred
    masked_pred = masked_pred * mask
    masked_gt = masked_gt * mask

    # bring to shape bz * ch * H * W
    masked_pred = masked_pred.permute(0, 3, 1, 2)
    masked_gt = masked_gt.permute(0, 3, 1, 2)
    mask = mask.permute(0, 3, 1, 2)

    return masked_gt, masked_pred, mask


@torch.no_grad()
def patched_flare_inference(
    views,
    mesh: Mesh,
    FLAMEServer: FLAME,
    deformer_net: ForwardDeformer,
    renderer: Renderer,
    channels_gbuffer: list[str],
    lgt: light.EnvironmentLight,
    shader_path: str,
    ghostbone: bool = True,
    finetune_color: bool = True,
    new_albedo: Optional[torch.Tensor] = None,
    new_roughness: Optional[torch.Tensor] = None,
):
    patched_shader = NeuralShader.load(shader_path, device=device)
    patched_shader.eval()

    # copy from flare/modules/neuralshader.py
    def _forward(
        self, position, gbuffer, view_direction, mesh, light, deformed_position, skin_mask=None
    ):
        bz, h, w, ch = position.shape
        pe_input = self.apply_pe(position=position)

        view_dir = view_direction[:, None, None, :]
        normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh)

        # ==========================================================================================
        # Albedo ; roughness; specular intensity
        # ==========================================================================================
        all_tex = self.material_mlp(pe_input.view(-1, self.inp_size).to(torch.float32))
        kd = all_tex[..., :3].view(bz, h, w, ch)
        kr = all_tex[..., 3:4]
        kr = kr.view(bz, h, w, 1).to(self.device)
        ko = all_tex[..., 4:5]
        ko = ko.view(bz, h, w, 1)

        # inject new albedo and roughnes
        if new_albedo is not None:
            kd = new_albedo.to(self.device)

        if new_roughness is not None:
            kr = new_roughness.to(self.device)

        if skin_mask is not None:
            fresnel_constant = torch.ones((bz, h, w, 1)).to(self.device) * 0.047
            fresnel_constant[skin_mask] = 0.028
        else:
            fresnel_constant = 0.04

        # ========= diffuse shading ===========
        kr_max = torch.ones((bz, h, w, 1))
        kr_max = kr_max.to(self.device)
        enc_nd_kr_max = self.dir_enc_func(normal_bend.view(-1, 3), kr_max.view(-1, 1))
        shading = self.light_mlp(enc_nd_kr_max)
        shading = shading.view(bz, h, w, 3)

        # ========= specular shading shading ===========
        color, buffers = light.shade_pbr_ipe(
            deformed_position,
            shading,
            self.dir_enc_func,
            self.light_mlp,
            normal_bend,
            kd,
            kr,
            view_dir,
            ko,
            normal_bend,
            fresnel_constant,
        )

        return color, kd, buffers

    patched_shader.forward = types.MethodType(_forward, patched_shader)

    return run_inference(
        views,
        mesh,
        FLAMEServer,
        deformer_net,
        patched_shader,
        renderer,
        channels_gbuffer,
        lgt,
        ghostbone,
        finetune_color,
    )


def to_masked_rgba(rgb_pred, mask1c):
    bz, H, W, _ = rgb_pred.shape
    return torch.lerp(
        torch.zeros((bz, H, W, 4)),
        torch.concat([rgb_pred.cpu(), torch.ones_like(rgb_pred[..., 0:1]).cpu()], axis=3),
        mask1c.float(),
    )
