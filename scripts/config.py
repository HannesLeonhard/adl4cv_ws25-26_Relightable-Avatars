from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass
import json
from typing import Any


@dataclass
class PathConfig:
    _working_dir: str = field(repr=False)
    run_name: str
    input_dir: str
    train_dir: list[str]
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
    
    def images_save_path(self, stage: str) -> Path:
        dir = self.experiment_dir / stage / "images"
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir
    
    def images_eval_path(self) -> Path:
        dir = self.experiment_dir / "images_evaluation"
        if not dir.exists():
            dir.mkdir(parents=True)
        return dir

@dataclass
class MaterialAwareTrainingConfig:
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
    weight_roughness_regularization: float = 0.1
    weight_white_lgt_regularization: float = 1.0
    weight_fresnel_coeff: float = 0.01
    weight_diffusion_albedo_regularization: float = 0.1
    weight_diffusion_normal_regularization: float = 0.1
    weight_diffusion_roughness_regularization: float = 0.1
    weight_diffusion_irradiance_regularization: float = 0.0

    bsdf: str = "pbr_shading"
    activation: str = "relu"
    fourier_features: str = "positional"
    light_mlp_ch: int = 3
    light_mlp_dims: list[int] = field(default_factory=lambda: [64, 64])
    material_mlp_ch: int = 5
    material_mlp_dims: list[int] = field(default_factory=lambda: [128, 128, 128, 128, 128])
    r_mean: float = 0.5

    ghostbone: bool = True
    deform_dims: list[int] = field(default_factory=lambda: [128, 128, 128, 128])

    visualization_frequency: int = 300
    save_frequency: int = 0
    visualization_views: list[int] = field(default_factory=lambda: [15, 25, 27, 21, 26])

    @classmethod
    def default_stage_1_config(cls, batch_size: int, ghostbone: bool):
        return cls(
            finetune_color=False,
            train_deformer=True,
            batch_size=batch_size,
            iterations=1500,
            ghostbone=ghostbone,
        )

    @classmethod
    def default_stage_2_config(cls, batch_size: int, ghostbone: bool):
        return cls(
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


def serialize_dataclass_to_dict(obj: Any) -> Any:
    """Recursively converts a dataclass object into a dictionary suitable for JSON."""
    # dataclass -> use asdict to get field values (recursively handled below)
    if is_dataclass(obj):
        data = asdict(obj)
        return {k: serialize_dataclass_to_dict(v) for k, v in data.items()}

    # pathlib.Path (including PosixPath/WindowsPath)
    if isinstance(obj, Path):
        # return POSIX-style path (always uses forward slashes)
        return obj.as_posix()

    # dict -> serialize keys and values (keys must become strings for JSON)
    if isinstance(obj, dict):
        return {str(serialize_dataclass_to_dict(k)): serialize_dataclass_to_dict(v) for k, v in obj.items()}

    # sequences -> list
    if isinstance(obj, (list, tuple, set)):
        return [serialize_dataclass_to_dict(item) for item in obj]

    # primitive JSON-serializable types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # fallback: try to return a JSON-friendly representation
    # prefer to return the object itself if json can handle it; otherwise return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def write_config_to_json(
    path_config: PathConfig,
    train_config: MaterialAwareTrainingConfig,
    file_path: Path
):
    """
    Combines two dataclass configurations into a single dictionary and writes
    it to a JSON file.
    """
    # Create the top-level configuration dictionary
    config_data = {
        "PathConfig": serialize_dataclass_to_dict(path_config),
        "MaterialAwareTrainingConfig": serialize_dataclass_to_dict(train_config),
    }

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the dictionary to the JSON file
    with open(file_path, "w") as f:
        # Use indent for human-readable output
        json.dump(config_data, f, indent=4)
        
    print(f"Configuration successfully written to: {file_path.resolve()}")

