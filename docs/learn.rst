Learn (Examples)
================

``fastrad`` includes a dedicated ``examples/`` directory within the repository demonstrating extensive usage patterns mapping natively onto public clinical datasets (such as the TCIA RIDER Lung CT open cohort).

01: Basic Feature Extraction
----------------------------
The core workflow for loading medical volumes (NIfTI/NRRD/DICOM), converting physical geometries into tensor arrays via SimpleITK, configuring the extractor settings, and evaluating mathematically compliant output dictionaries.
(`View Source in Repository <https://github.com/radiomicslab/fastrad/blob/main/examples/01_basic_extraction.py>`_)

02: PyTorch CUDA GPU Acceleration Benchmark
-------------------------------------------
A comprehensive script documenting how to offload evaluation streams securely into CUDA PyTorch frameworks. Benchmarking native synthetic 128x128x128 clinical matrices to demonstrate the monumental evaluation latency differences between parallel hardware routing vs single-threaded environments.
(`View Source in Repository <https://github.com/radiomicslab/fastrad/blob/main/examples/02_gpu_acceleration.py>`_)

03: Multi-Patient Clinical Batch Processing
-------------------------------------------
Simulating automated high-throughput workflows executing standard IBSI evaluations identically mapped onto multiple patient modalities concurrently. Displays how a singular ``FeatureExtractor`` efficiently intercepts multiple sequential evaluation queries cleanly.
(`View Source in Repository <https://github.com/radiomicslab/fastrad/blob/main/examples/03_batch_processing_tcia.py>`_)

04: Advanced Configurations & Fallback Logic
--------------------------------------------
Demonstrating how `fastrad` specifically navigates unstable runtime bounds (anisotropic geometric distortion signals, OutOfMemory GPU triggers, and internal tensor generation exceptions) dynamically while guaranteeing uncorrupted local features.
(`View Source in Repository <https://github.com/radiomicslab/fastrad/blob/main/examples/04_advanced_configuration.py>`_)
