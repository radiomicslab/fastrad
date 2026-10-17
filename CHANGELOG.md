# Changelog

## Unreleased (performance and parity)

### Fixed
- **Shape parity with PyRadiomics.** The marching-cubes mesh is now built in
  PyRadiomics' coordinate frame (mask cropped to its bounding box, padded by one
  voxel). The mesh produced by PyRadiomics' tables is not perfectly closed on
  irregular boundaries (in PyRadiomics too), and the signed volume of an open
  mesh depends on the coordinate origin, so `mesh_volume` and the features
  derived from it (`sphericity`, `surface_volume_ratio`, compactness,
  spherical disproportion) previously differed from PyRadiomics by up to
  hundreds of mm³ on irregular ROIs. They now match exactly.
- `FeatureSettings()` with the default `device="auto"` crashed in shape
  extraction.

### Performance
- Image and mask are cropped to the ROI bounding box (+1 voxel) on the host
  before being moved to the device. Texture matrices are unchanged by
  construction; memory and work no longer scale with the full image size.
- One device-to-host copy for all feature values instead of one per feature.
- Shape: all triangles processed in one batch; diameters computed on the host
  with NumPy (removes ~575k small tensor operations on a typical tumour).
- GLRLM: run lengths for all 13 directions from one sort (removes a per-voxel
  stepping loop that synchronised on every step).
- GLCM: ROI coordinates computed once; all 13 matrices from one `bincount`;
  MCC eigenproblems solved on the CPU.
- GLSZM: grey levels labelled in batched single calls instead of one call per
  level; `fastrad.features.glszm.LAST_LABEL_BACKEND` records whether CuPy or
  SciPy was used.

### Tests and benchmarks
- `tests/synthetic.py` and `tests/test_pyradiomics_parity.py`: parity with
  PyRadiomics on synthetic CT-like cases with irregular ROIs, no downloads needed.
- `benchmarks/run_runtime_performance.py`: end-to-end figures are single full
  extraction calls (previously sums of per-class calls, which counted
  PyRadiomics' fixed per-call overhead once per class).
- `benchmarks/run_single_call_latency.py`: quick like-for-like latency check.
- `benchmarks/run_cohort_throughput.py`: thread pinning per worker,
  `--replicate`, `--fastrad-cpu-workers`, crash retry, incremental saving.
