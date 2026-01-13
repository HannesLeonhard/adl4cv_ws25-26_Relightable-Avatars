from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    _working_dir: str = field(repr=False)
    run_name: str
    input_dir: str
    train_dir: list[str]
    eval_dir: list[str]
    output_dir: str
    flame_path: str
    diffusion_dir: str

    @property
    def working_dir(self) -> Path:
        return Path(self._working_dir)

    @property
    def experiment_dir(self) -> Path:
        dir = Path(self.working_dir) / self.output_dir / self.run_name
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir
    
    def image_save_path(self, stage: str) -> Path:
        dir = self.experiment_dir / stage / "images"
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir
    
    def image_eval_save_path(self, subfolder: str) -> Path:
        dir = self.experiment_dir / "images_evaluation" / subfolder
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir

    def meshes_save_path(self, stage: str) -> Path:
        dir = self.experiment_dir / stage / "meshes"
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir

    def shaders_save_path(self, stage: str) -> Path:
        dir = self.experiment_dir / stage / "network_weights"
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir


@dataclass
class MaterialAwareTrainingConfig:
    stage: str
    finetune_color: bool
    train_deformer: bool

    batch_size: int
    iterations: int
    upsample_iterations: list[int] = field(default_factory=lambda: [500])
    sample_idx_ratio: int = 1
    downsample: bool = False
    downsample_ratio: float = 0.03
    grad_scale: bool = False

    decay_flame: list[int] = field(default_factory=lambda: [100])
    flame_mask: bool = False

    lr_vertices: float = 1e-3
    lr_shader: float = 1e-3
    lr_deformer: float = 1e-3

    weight_mask: float = 2.0
    weight_normal: float = 0.1
    weight_laplacian: float = 60.0
    weight_shading: float = 1.0
    weight_perceptual_loss: float = 0.1
    weight_flame_regularization: float = 10.0
    weight_albedo_regularization: float = 0.01
    weight_roughness_regularization: float = 0.01
    weight_white_lgt_regularization: float = 0.01
    weight_fresnel_coeff: float = 0.01
    weight_diffusion_albedo_regularization: float = 0.0
    weight_diffusion_normal_regularization: float = 0.0
    weight_diffusion_roughness_regularization: float = 0.0
    weight_diffusion_irradiance_regularization: float = 0.0

    bsdf: str = "pbr_shading"
    activation: str = "relu"
    fourier_features: str = "positional"
    light_mlp_ch: int = 3
    light_mlp_dims: list[int] = field(default_factory=lambda: [64, 64])
    material_mlp_ch: int = 5
    material_mlp_dims: list[int] = field(default_factory=lambda: [128, 128, 128, 128])
    r_mean: float = 0.5

    ghostbone: bool = True
    deform_dims: list[int] = field(default_factory=lambda: [128, 128, 128, 128])

    visualization_frequency: int = 100
    save_frequency: int = 300
    visualization_views: list[int] = field(default_factory=lambda: [15, 25, 27, 21, 26])
    

    @classmethod
    def default_stage_1_config(cls, batch_size: int, ghostbone: bool):
        return cls(
            stage="stage_1",
            finetune_color=False,
            train_deformer=True,
            batch_size=batch_size,
            iterations=1500,
            ghostbone=ghostbone,
        )

    @classmethod
    def default_stage_2_config(cls, batch_size: int, ghostbone: bool):
        return cls(
            stage="stage_2",
            finetune_color=True,
            train_deformer=False,
            batch_size=batch_size,
            iterations=1000,
            fourier_features="hashgrid",
            material_mlp_dims=[64, 64],
            light_mlp_dims=[64, 64],
            lr_vertices=1e-5,
            ghostbone=ghostbone,
        )
