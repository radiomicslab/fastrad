Installation
============

You can install `fastrad` and its core dependencies easily using standard Python package managers. We recommend using a virtual environment (e.g., `venv`, `conda`, or `uv`).

Prerequisites
-------------
- Python 3.11+
- A compatible PyTorch installation (if using hardware acceleration)

Installing from Source
----------------------
Currently, `fastrad` is actively developed on GitHub. To install the latest stable version:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/radiomicslab/fastrad.git
   cd fastrad

   # Install standard CPU dependencies
   pip install .

Hardware Acceleration (CUDA)
----------------------------
For NVIDIA GPU hardware acceleration, which unlocks the massive 10x-70x speedups, you should ensure that your environment has a CUDA-compatible version of PyTorch installed.

To install with CUDA-specific optional dependencies:

.. code-block:: bash

   pip install ".[cuda]"

cuCIM for GLSZM Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Gray Level Size Zone Matrix (GLSZM) relies on Connected-Component labeling. `fastrad` uses a hyper-optimized hybrid pipeline for this. 

If running on CPU, `fastrad` uses the highly optimized C-backed `scipy.ndimage.label`. However, if you are executing on a CUDA GPU, the **cuCIM** (RAPIDS) dependency is automatically installed via the `[cuda]` extension, allowing `fastrad` to natively keep the connected-component computations on the GPU without memory transfer penalties.
