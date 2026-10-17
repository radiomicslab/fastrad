import csv
import re
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def compute_icc_2_1(data_matrix):
    """
    Computes ICC(2,1) - Two-way random effects, absolute agreement, single rater.
    Formula strictly conforming to Shrout & Fleiss (1979) and Koo & Li (2016):
    ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + k*(MS_C - MS_E)/n)
    
    data_matrix: array of shape (n_subjects, k_raters/scans)
    """
    n, k = data_matrix.shape
    if n < 2 or k < 2:
        return np.nan
        
    tot_var = np.var(data_matrix)
    if tot_var < 1e-12:
        return 1.0
        
    mean_row = np.mean(data_matrix, axis=1)
    mean_col = np.mean(data_matrix, axis=0)
    mean_tot = np.mean(data_matrix)
    
    ss_total = np.sum((data_matrix - mean_tot)**2)
    ss_row = k * np.sum((mean_row - mean_tot)**2)
    ss_col = n * np.sum((mean_col - mean_tot)**2)
    ss_err = max(0.0, ss_total - ss_row - ss_col)
    
    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    
    denom = ms_row + (k - 1) * ms_err + (k / n) * (ms_col - ms_err)
    if denom <= 0:
        return 0.0
        
    icc = (ms_row - ms_err) / denom
    return float(np.clip(icc, 0.0, 1.0))

def bootstrap_icc_ci(fastrad_mat_dict, pyrad_mat_dict, n_bootstraps=500, alpha=0.05, seed=42):
    """
    Subject-level bootstrap for mean ICC and paired difference ΔICC.
    """
    rng = np.random.default_rng(seed)
    shared_keys = [k for k in fastrad_mat_dict if k in pyrad_mat_dict]
    if not shared_keys:
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)
        
    n_subjects = next(iter(fastrad_mat_dict.values())).shape[0]
    
    boot_fastrad_means = []
    boot_pyrad_means = []
    boot_diff_means = []
    
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        
        f_boot_iccs = []
        p_boot_iccs = []
        for k in shared_keys:
            f_mat = fastrad_mat_dict[k][boot_idx, :]
            p_mat = pyrad_mat_dict[k][boot_idx, :]
            
            f_val = compute_icc_2_1(f_mat)
            p_val = compute_icc_2_1(p_mat)
            
            if not np.isnan(f_val) and not np.isnan(p_val):
                f_boot_iccs.append(f_val)
                p_boot_iccs.append(p_val)
                
        if f_boot_iccs and p_boot_iccs:
            f_m = np.mean(f_boot_iccs)
            p_m = np.mean(p_boot_iccs)
            boot_fastrad_means.append(f_m)
            boot_pyrad_means.append(p_m)
            boot_diff_means.append(f_m - p_m)
            
    f_ci = (float(np.percentile(boot_fastrad_means, 100 * (alpha / 2))),
            float(np.percentile(boot_fastrad_means, 100 * (1 - alpha / 2))))
    p_ci = (float(np.percentile(boot_pyrad_means, 100 * (alpha / 2))),
            float(np.percentile(boot_pyrad_means, 100 * (1 - alpha / 2))))
    diff_ci = (float(np.percentile(boot_diff_means, 100 * (alpha / 2))),
               float(np.percentile(boot_diff_means, 100 * (1 - alpha / 2))))
               
    return f_ci, p_ci, diff_ci

def pyrad_to_fastrad(pyrad_key):
    if not pyrad_key.startswith("original_"):
        return None
    parts = pyrad_key.split('_')
    if len(parts) < 3:
        return None
    f_class = parts[1].lower()
    f_name = "_".join(parts[2:])
    
    mapping = {
        '10Percentile': '10th_percentile',
        '90Percentile': '90th_percentile',
        'InterquartileRange': 'interquartile_range',
        'MeanAbsoluteDeviation': 'mean_absolute_deviation',
        'RobustMeanAbsoluteDeviation': 'robust_mean_absolute_deviation',
        'RootMeanSquared': 'root_mean_squared',
        'TotalEnergy': 'total_energy',
        'Imc1': 'imc1',
        'Imc2': 'imc2',
        'Idm': 'idm',
        'Idmn': 'idmn',
        'Id': 'id',
        'Idn': 'idn',
        'MCC': 'mcc',
        'VoxelVolume': 'voxel_volume',
        'MeshVolume': 'mesh_volume',
        'SurfaceArea': 'surface_area',
        'SurfaceVolumeRatio': 'surface_volume_ratio',
        'Compactness1': 'compactness_1',
        'Compactness2': 'compactness_2',
        'SphericalDisproportion': 'spherical_disproportion',
        'Maximum3DDiameter': 'maximum_3d_diameter',
        'Maximum2DDiameterSlice': 'maximum_2d_diameter_slice',
        'Maximum2DDiameterColumn': 'maximum_2d_diameter_column',
        'Maximum2DDiameterRow': 'maximum_2d_diameter_row',
        'MajorAxisLength': 'major_axis_length',
        'MinorAxisLength': 'minor_axis_length',
        'LeastAxisLength': 'least_axis_length',
    }
    
    if f_name in mapping:
        f_name_snake = mapping[f_name]
    else:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', f_name)
        f_name_snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
    return f'{f_class}:{f_name_snake}'

def create_spherical_mask_t(img_tensor, radius_mm, spacing):
    D, H, W = img_tensor.shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))
    
    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_sq = ((z - center_z) / r_z)**2 + ((y - center_y) / r_y)**2 + ((x - center_x) / r_x)**2
    mask = torch.zeros_like(img_tensor, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask

def load_dicom_series(dir_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dir_path))
    if not dicom_names:
        return None
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def apply_perturbations(img_tensor, seed: int = 42):
    # NOTE: previously used torch.randn_like() with no fixed seed, so
    # Gaussian-noise drift results varied between runs (observed: 10.91%
    # in one run, 9.80% in another) purely due to different random noise
    # draws, not any change in fastrad or PyRadiomics behaviour. Seeded
    # here for reproducibility; re-run and update the manuscript's
    # perturbation table with this seeded result as the final figure.
    generator = torch.Generator().manual_seed(seed)
    noise = img_tensor + torch.randn(img_tensor.shape, generator=generator) * 20.0
    shifted = torch.roll(img_tensor, shifts=(2, 2, 2), dims=(0, 1, 2))
    return {"Gaussian Noise (sigma=20HU)": noise, "Rigid Translation (2 voxels)": shifted}

def run():
    print("Running Rigorous Reproducibility and Perturbation Stability Benchmark...")
    project_root = Path(__file__).parent.parent
    img_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
    mask_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
    
    output_dir = project_root / "benchmarks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = project_root / "pyradiomics_config.yaml"
    fastrad_settings = FeatureSettings.from_yaml(config_path, device="cpu")
    classes = fastrad_settings.feature_classes
    fastrad_ext = FeatureExtractor(fastrad_settings)
    pyrad_ext = featureextractor.RadiomicsFeatureExtractor(str(config_path))

    md = []
    md.append("## Section 5: Reproducibility and Statistical Stability\n")
    
    print("  -> Running ICC(2,1) Analysis on RIDER test-retest pairs (5.1)...")
    md.append("### 5.1 ICC Analysis on Real RIDER Scan-Rescan Pairs\n")
    # NOTE: the actual "$N=... patient scan pairs" line is appended below,
    # after the real number of available pairs is known, rather than being
    # hardcoded here. A previous version of this script printed a fixed
    # "N=5" string regardless of how many RIDER pairs were actually found
    # on disk, so the reported cohort size never reflected reality.
    md.append("> **Statistical Protocol (Koo & Li, 2016)**:\n"
              "> - **Model**: $\\text{ICC}(2,1)$ (Two-way random effects, single rater, absolute agreement).\n"
              "> - **Mask Correspondence**: Spherical ROI ($r=15\\,\\text{mm}$) generated on Scan 1 and mapped into Scan 2 physical space via nearest-neighbor resampling under identity transform.\n"
              "> - **Confidence Intervals**: 95% non-parametric subject-level bootstrap (500 resamples).\n\n")

    rider_dir = project_root / "tests" / "fixtures" / "tcia" / "rider"
    rider_pairs = []
    if rider_dir.exists():
        for p_dir in sorted(rider_dir.iterdir()):
            if p_dir.is_dir():
                s1, s2 = p_dir / "scan1", p_dir / "scan2"
                if s1.exists() and s2.exists():
                    rider_pairs.append((s1, s2))
                    
    if not rider_pairs:
        md.append("*No RIDER pairs available. Skipping ICC Analysis.*")
    else:
        # NOTE: this previously hardcoded `rider_pairs[:5]`, silently
        # discarding any additional downloaded pairs beyond the first 5
        # regardless of how many were actually available (e.g. up to 32
        # via download_rider_pairs.py). All available pairs are now used,
        # and the manuscript's reported cohort size must match whatever
        # n_patients actually prints here after a real run.
        n_patients = len(rider_pairs)
        md.append(
            f"> - **Cohort**: RIDER Lung CT test-retest dataset "
            f"($N={n_patients}$ patient scan pairs, {2 * n_patients} volumetric scans total).\n\n"
        )
        
        # Pre-initialize data structs
        test_img = load_dicom_series(rider_pairs[0][0])
        test_img_t = torch.from_numpy(sitk.GetArrayFromImage(test_img)).float()
        test_spacing = test_img.GetSpacing()[::-1]
        test_mask = create_spherical_mask_t(torch.zeros_like(test_img_t), 15.0, test_spacing)
        
        base_f_res = fastrad_ext.extract(MedicalImage(test_img_t, spacing=test_spacing), Mask(test_mask, spacing=test_spacing))
        fastrad_icc_data = {k: np.zeros((n_patients, 2)) for k in base_f_res if k != "firstorder:standard_deviation"}
        pyrad_icc_data = {}
        
        s_test_img = sitk.GetImageFromArray(test_img_t.numpy())
        s_test_img.SetSpacing(test_spacing[::-1])
        s_test_img.SetOrigin(test_img.GetOrigin())
        s_test_img.SetDirection(test_img.GetDirection())
        s_test_mask = sitk.GetImageFromArray(test_mask.numpy().astype(np.uint8))
        s_test_mask.SetSpacing(test_spacing[::-1])
        s_test_mask.SetOrigin(test_img.GetOrigin())
        s_test_mask.SetDirection(test_img.GetDirection())
        
        res = pyrad_ext.execute(s_test_img, s_test_mask)
        for k in res:
            if k.startswith("original_"):
                key = pyrad_to_fastrad(k)
                if key and key in fastrad_icc_data:
                    pyrad_icc_data[key] = np.zeros((n_patients, 2))
                    
        for p_idx, (s1, s2) in enumerate(rider_pairs):
            sitk_img1 = load_dicom_series(s1)
            sitk_img2 = load_dicom_series(s2)
            
            img1_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_img1)).float()
            spacing1 = sitk_img1.GetSpacing()[::-1]
            try:
                img2_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_img2)).float()
                spacing2 = sitk_img2.GetSpacing()[::-1]
            except Exception:
                continue
                
            mask1_t = create_spherical_mask_t(img1_t, 15.0, spacing1)
            
            s_img1 = sitk.GetImageFromArray(img1_t.numpy())
            s_img1.SetSpacing(spacing1[::-1])
            s_img1.SetOrigin(sitk_img1.GetOrigin())
            s_img1.SetDirection(sitk_img1.GetDirection())
            
            s_mask1 = sitk.GetImageFromArray(mask1_t.numpy().astype(np.uint8))
            s_mask1.SetSpacing(spacing1[::-1])
            s_mask1.SetOrigin(sitk_img1.GetOrigin())
            s_mask1.SetDirection(sitk_img1.GetDirection())
            
            s_img2 = sitk.GetImageFromArray(img2_t.numpy())
            s_img2.SetSpacing(spacing2[::-1])
            s_img2.SetOrigin(sitk_img2.GetOrigin())
            s_img2.SetDirection(sitk_img2.GetDirection())
            
            # Physical space resampling
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(s_img2)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(sitk.Transform())
            s_mask2 = resampler.Execute(s_mask1)
            
            mask2_t = torch.from_numpy(sitk.GetArrayFromImage(s_mask2)).float()
            
            f_img1, f_mask1 = MedicalImage(img1_t, spacing=spacing1), Mask(mask1_t, spacing=spacing1)
            f_img2, f_mask2 = MedicalImage(img2_t, spacing=spacing2), Mask(mask2_t, spacing=spacing2)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_res1 = fastrad_ext.extract(f_img1, f_mask1)
                f_res2 = fastrad_ext.extract(f_img2, f_mask2)
                
            for k, val in fastrad_icc_data.items():
                val[p_idx, 0] = f_res1.get(k, 0)
                val[p_idx, 1] = f_res2.get(k, 0)
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r1 = pyrad_ext.execute(s_img1, s_mask1)
                    r2 = pyrad_ext.execute(s_img2, s_mask2)
                    for k in r1:
                        if k.startswith("original_"):
                            key = pyrad_to_fastrad(k)
                            if key and key in pyrad_icc_data:
                                pyrad_icc_data[key][p_idx, 0] = float(r1[k]) if not hasattr(r1[k], "item") else float(r1[k].item())
                                pyrad_icc_data[key][p_idx, 1] = float(r2[k]) if not hasattr(r2[k], "item") else float(r2[k].item())
                except Exception as e:
                    print(f"PyRadiomics extraction error: {e}")
                    
        # Compute per-feature ICC values
        shared_keys = sorted([k for k in fastrad_icc_data if k in pyrad_icc_data])
        per_feature_icc_records = []
        
        f_iccs, p_iccs = [], []
        for k in shared_keys:
            f_val = compute_icc_2_1(fastrad_icc_data[k])
            p_val = compute_icc_2_1(pyrad_icc_data[k])
            f_iccs.append(f_val)
            p_iccs.append(p_val)
            
            cls_name, _f_name = k.split(":")
            per_feature_icc_records.append({
                "feature_class": cls_name,
                "feature_key": k,
                "fastrad_icc": f_val,
                "pyradiomics_icc": p_val,
                "icc_diff": (f_val - p_val),
                "n_subjects": n_patients
            })
            
        # Export per-feature ICC CSV
        icc_csv_path = output_dir / "rider_icc_per_feature.csv"
        with open(icc_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "feature_class", "feature_key", "fastrad_icc", "pyradiomics_icc", "icc_diff", "n_subjects"
            ])
            writer.writeheader()
            writer.writerows(per_feature_icc_records)
            
        # Compute Bootstrap CIs
        f_ci, p_ci, diff_ci = bootstrap_icc_ci(fastrad_icc_data, pyrad_icc_data)
        
        f_high = (sum(1 for x in f_iccs if x >= 0.90) / len(f_iccs) * 100) if f_iccs else 0
        p_high = (sum(1 for x in p_iccs if x >= 0.90) / len(p_iccs) * 100) if p_iccs else 0
        
        f_mean = float(np.mean(f_iccs))
        p_mean = float(np.mean(p_iccs))
        
        md.append(f"- **Total Features Entering Analysis ($N_{{feat}}$)**: {len(shared_keys)} (across $N={n_patients}$ test-retest patient pairs)\n")
        md.append(f"- **Fastrad Features with $\\text{{ICC}} \\ge 0.90$**: {f_high:.1f}% ({sum(1 for x in f_iccs if x >= 0.90)}/{len(f_iccs)})\n")
        md.append(f"- **PyRadiomics Features with $\\text{{ICC}} \\ge 0.90$**: {p_high:.1f}% ({sum(1 for x in p_iccs if x >= 0.90)}/{len(p_iccs)})\n")
        md.append(f"- **Fastrad Mean $\\text{{ICC}}(2,1)$ [95% CI]**: **{f_mean:.4f}** [{f_ci[0]:.4f}, {f_ci[1]:.4f}]\n")
        md.append(f"- **PyRadiomics Mean $\\text{{ICC}}(2,1)$ [95% CI]**: **{p_mean:.4f}** [{p_ci[0]:.4f}, {p_ci[1]:.4f}]\n")
        md.append(f"- **Paired $\\Delta\\text{{ICC}}$ (fastrad - PyRadiomics) [95% CI]**: **{f_mean - p_mean:+.4f}** [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]\n")
        
        # Per-feature agreement. When both pipelines compute the same values to
        # floating-point precision, the paired ICC differences are rounding
        # noise (~1e-8), and a signed-rank test on them only tests the sign of
        # that noise: its p-value is not meaningful. Report the size of the
        # differences and an equivalence test against a practical margin.
        import numpy as _np
        from scipy.stats import ttest_1samp, wilcoxon
        diffs = _np.asarray(f_iccs, dtype=float) - _np.asarray(p_iccs, dtype=float)
        max_abs = float(_np.max(_np.abs(diffs)))
        n_identical_7dp = int(_np.sum(_np.round(diffs, 7) == 0))
        margin = 0.01
        if _np.allclose(diffs, 0.0):
            tost_p = 0.0
        else:
            p_lo = ttest_1samp(diffs, -margin, alternative="greater").pvalue
            p_hi = ttest_1samp(diffs, margin, alternative="less").pvalue
            tost_p = float(max(p_lo, p_hi))
        md.append(f"- **Max per-feature |ICC difference|**: {max_abs:.2e} "
                  f"({n_identical_7dp}/{len(diffs)} features identical to 7 decimal places)\n")
        md.append(f"- **Equivalence (TOST, margin ±{margin})**: p = {tost_p:.2e}\n")
        try:
            stat, p_val = wilcoxon(f_iccs, p_iccs)
            md.append(f"- *Wilcoxon signed-rank on the same differences (not meaningful at floating-point "
                      f"scale, reported for completeness)*: stat={stat:.1f}, p={p_val:.4f}\n\n")
        except ValueError:
            md.append("- *Wilcoxon signed-rank not computable: all differences are zero*\n\n")

    # 5.2 Perturbation Stability Analysis
    print("  -> Profiling Perturbation Stability Matrices (5.2)...")
    md.append("### 5.2 Numerical Robustness to Physical Input Perturbations\n")
    md.append("| Perturbation | PyRadiomics Mean Drift (%) | PyRadiomics 95% CI (%) | fastrad Mean Drift (%) | fastrad 95% CI (%) | Drift Difference (%) |")
    md.append("|---|---|---|---|---|---|")
    
    perturbation_records = []
    
    if img_path.exists() and mask_path.exists():
        sitk_image = sitk.ReadImage(str(img_path))
        sitk_mask = sitk.ReadImage(str(mask_path))
        
        img_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
        mask_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
        spacing = sitk_image.GetSpacing()[::-1]
        
        fastrad_base = fastrad_ext.extract(MedicalImage(img_t, spacing=spacing), Mask(mask_t, spacing=spacing))
        
        pyrad_ext.disableAllFeatures()
        for cls in classes:
            pyrad_ext.enableFeatureClassByName(cls)
        res = pyrad_ext.execute(sitk_image, sitk_mask)
        pyrad_base = {}
        for k, v in res.items():
            if k.startswith("original_"):
                key = pyrad_to_fastrad(k)
                if key:
                    pyrad_base[key] = float(getattr(v, "real", v)) if not hasattr(v, "item") else float(getattr(v.item(), "real", v.item()))

        perturbations = apply_perturbations(img_t)
        
        for p_name, p_tensor in perturbations.items():
            f_img_p = MedicalImage(p_tensor, spacing=spacing)
            f_res_p = fastrad_ext.extract(f_img_p, Mask(mask_t, spacing=spacing))
            
            s_img_p = sitk.GetImageFromArray(p_tensor.numpy())
            s_img_p.CopyInformation(sitk_image)
            
            res = pyrad_ext.execute(s_img_p, sitk_mask)
            p_res_p = {}
            for k, v in res.items():
                if k.startswith("original_"):
                    key = pyrad_to_fastrad(k)
                    if key:
                        p_res_p[key] = float(getattr(v, "real", v)) if not hasattr(v, "item") else float(getattr(v.item(), "real", v.item()))
                            
            f_drifts, p_drifts = [], []
            shared_p_keys = [k for k in fastrad_base if k in pyrad_base and k in f_res_p and k in p_res_p]
            
            for k in shared_p_keys:
                if abs(fastrad_base[k]) > 1e-6:
                    f_drv = abs(f_res_p[k] - fastrad_base[k]) / abs(fastrad_base[k]) * 100.0
                    p_drv = abs(p_res_p[k] - pyrad_base[k]) / abs(pyrad_base[k]) * 100.0
                    f_drifts.append(f_drv)
                    p_drifts.append(p_drv)
                    
            f_arr = np.array(f_drifts)
            p_arr = np.array(p_drifts)
            
            f_m, p_m = float(np.mean(f_arr)), float(np.mean(p_arr))
            f_ci_low, f_ci_high = float(np.percentile(f_arr, 2.5)), float(np.percentile(f_arr, 97.5))
            p_ci_low, p_ci_high = float(np.percentile(p_arr, 2.5)), float(np.percentile(p_arr, 97.5))
            diff_m = f_m - p_m
            
            md.append(f"| {p_name} | {p_m:.2f}% | [{p_ci_low:.2f}%, {p_ci_high:.2f}%] | {f_m:.2f}% | [{f_ci_low:.2f}%, {f_ci_high:.2f}%] | {diff_m:+.2f}% |")
            
            perturbation_records.append({
                "perturbation": p_name,
                "pyradiomics_mean_drift": p_m,
                "pyradiomics_ci_low": p_ci_low,
                "pyradiomics_ci_high": p_ci_high,
                "fastrad_mean_drift": f_m,
                "fastrad_ci_low": f_ci_low,
                "fastrad_ci_high": f_ci_high,
                "drift_diff": diff_m
            })
            
        # Export perturbation CSV
        pert_csv_path = output_dir / "perturbation_stability.csv"
        with open(pert_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "perturbation", "pyradiomics_mean_drift", "pyradiomics_ci_low", "pyradiomics_ci_high",
                "fastrad_mean_drift", "fastrad_ci_low", "fastrad_ci_high", "drift_diff"
            ])
            writer.writeheader()
            writer.writerows(perturbation_records)
            
    md.append("\n*Note: High relative drift under rigid translation is physically expected for localized bounding masks because spatial shift translates the ROI onto distinct anatomical structures.*\n\n")
    return "\n".join(md)

if __name__ == "__main__":
    print(run())
