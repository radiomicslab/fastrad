import time
from pathlib import Path

import torch

from fastrad import DenseFeatureExtractor, FeatureSettings, Mask, MedicalImage


def run():
    print("Running Dense Voxel-Wise Hardware Extraction Performance Benchmark...")
    project_root = Path(__file__).parent.parent
    config_path = project_root / "pyradiomics_config.yaml"
    md = []
    md.append("## Section 4: Dense Voxel-Wise Hardware Extraction Performance\n")
    md.append(
        "This section evaluates the runtime extraction performance scaling of `fastrad` when evaluating sliding windows densely across a large clinical tissue volume block, producing explicit multi-channel natively tracked spatial PyTorch Tensor maps instead of single scalar representations.\n"
    )

    # Generate 64x64x64 dummy uniform structural block
    D, H, W = 64, 64, 64
    img_vol = torch.randn((D, H, W), dtype=torch.float32)
    mask_vol = torch.ones((D, H, W), dtype=torch.float32)

    fastrad_image = MedicalImage(img_vol)
    fastrad_mask = Mask(mask_vol)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    settings = FeatureSettings.from_yaml(
        config_path, feature_classes=["firstorder", "glcm"], device=device
    )
    extractor = DenseFeatureExtractor(settings)

    md.append(f"**Hardware Evaluation Config**: {device.upper()}\n")
    md.append("Volume Profile: 64x64x64 Matrix (262,144 physical spatial locations)\n")

    md.append(
        "| Kernel Size (Voxel) | Stride | Result Shape Map | Window Evaluations Executed | Execution Time (s) |"
    )
    md.append("|---|---|---|---|---|")

    configurations = [(32, 16), (24, 8), (16, 4)]

    for kernel_size, stride in configurations:
        t0 = time.time()
        # Ensure cleanup
        if device == "cuda":
            torch.cuda.empty_cache()

        features = extractor.extract_dense(
            fastrad_image, fastrad_mask, kernel_size=kernel_size, stride=stride
        )

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()

        # Calculate shape
        out_dim = (64 - kernel_size) // stride + 1
        shape_repr = f"[{out_dim}x{out_dim}x{out_dim}]"
        window_count = out_dim**3

        exec_time = t1 - t0
        md.append(
            f"| {kernel_size}^3 | {stride} | {shape_repr} | {window_count} | {exec_time:.2f}s |"
        )

    md.append(
        "\n*Note: Output feature maps are evaluated strictly to valid mathematical patches exclusively. `DenseFeatureExtractor` prevents padding out of bound math pollution, executing highly deterministic memory-strided patch views via `PyTorch F.unfold` framework logic.*"
    )
    md.append("\n\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
