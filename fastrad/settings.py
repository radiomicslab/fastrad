from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_yaml_file(config_path: str | Path) -> dict[str, Any]:
    """Helper to load a YAML file using PyYAML, ruamel.yaml, or a basic parser."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    try:
        from ruamel.yaml import YAML

        yaml_parser = YAML(typ="safe")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml_parser.load(f) or {}
    except ImportError:
        pass

    # Basic fallback line parser for standard PyRadiomics yaml
    data: dict[str, Any] = {"setting": {}, "featureClass": {}}
    current_section = None
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if (
                not line.startswith(" ")
                and not line.startswith("\t")
                and line_str.endswith(":")
            ):
                current_section = line_str[:-1].strip()
                continue
            if current_section == "setting" and ":" in line_str:
                k, v = line_str.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    if v.lower() == "true":
                        val: Any = True
                    elif v.lower() == "false":
                        val = False
                    elif "." in v:
                        val = float(v)
                    else:
                        val = int(v)
                except ValueError:
                    val = v
                data["setting"][k] = val
            elif current_section in ("featureClass", "features"):
                cls_name = line_str.split(":")[0].strip()
                if cls_name:
                    data.setdefault("featureClass", {})[cls_name] = None
    return data


@dataclass
class FeatureSettings:
    """
    Configuration parameters regulating standard feature extraction pipelines.

    Attributes:
        feature_classes: List of feature groups to extract (e.g., ``["firstorder", "glcm"]``).
        bin_width: Radiometric discretization width applied before extracting textures.
        device: Target execution device (``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``).
        spacing: Real world physical spacing array `(z, y, x)` propagated downstream.
        force2D: Flag controlling slice-by-slice 2D calculation routing.
        force2Ddimension: Target dimension slice axis when projecting 3D volumes to 2D.
    """

    feature_classes: list[str] = field(
        default_factory=lambda: [
            "firstorder",
            "shape",
            "glcm",
            "glrlm",
            "glszm",
            "gldm",
            "ngtdm",
        ]
    )
    bin_width: float = 25.0
    device: str = "auto"
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    force2D: bool = False
    force2Ddimension: int = 0
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    amp: bool = False
    differentiable: bool = False

    @classmethod
    def from_yaml(cls, config_path: str | Path, **kwargs: Any) -> "FeatureSettings":
        """
        Loads FeatureSettings from a PyRadiomics-compatible YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.
            **kwargs: Overrides for any FeatureSettings fields (e.g. device, feature_classes).
        """
        data = _load_yaml_file(config_path)
        return cls.from_dict(data, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "FeatureSettings":
        """
        Loads FeatureSettings from a dictionary structure matching PyRadiomics YAML specs.
        """
        settings_dict = data.get("setting", data.get("settings", {})) or {}
        feature_classes_dict = data.get("featureClass", data.get("features", {})) or {}

        if (
            isinstance(feature_classes_dict, dict)
            and feature_classes_dict
            or isinstance(feature_classes_dict, list)
            and feature_classes_dict
        ):
            feature_classes = [k.lower() for k in feature_classes_dict]
        else:
            feature_classes = [
                "firstorder",
                "shape",
                "glcm",
                "glrlm",
                "glszm",
                "gldm",
                "ngtdm",
            ]

        bin_width = float(
            settings_dict.get("binWidth", settings_dict.get("bin_width", 25.0))
        )
        force2D = bool(settings_dict.get("force2D", False))
        force2Ddimension = int(settings_dict.get("force2Ddimension", 0))

        cfg: dict[str, Any] = {
            "feature_classes": feature_classes,
            "bin_width": bin_width,
            "force2D": force2D,
            "force2Ddimension": force2Ddimension,
        }
        cfg.update(kwargs)
        return cls(**cfg)
