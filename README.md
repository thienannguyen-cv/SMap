<h1 align="center">
  <a href="https://github.com/thienannguyen-cv/SMap">
    <img src="https://raw.githubusercontent.com/thienannguyen-cv/SMap/main/logo.png" width="1024">
  </a>
</h1><br>

# SMap: Spatial Mapping for Dynamic 3D Inference

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/smap-torch.svg)](https://badge.fury.io/py/smap-torch)
![Coverage](https://thienannguyen-cv.github.io/SMap/coverage.svg)
[![CI - Test](https://github.com/thienannguyen-cv/SMap/actions/workflows/ci.yml/badge.svg)](https://github.com/thienannguyen-cv/SMap/actions/workflows/ci.yml)

SMap (Spatial Mapping) is an open-source PyTorch library for spatial mapping based on 2D representations. It serves as the foundation for the **"Dynamic 3D Inference"** vision, which explores a fundamental question:

> *"How to infer 3D properties of objects by moving their projection on a 2D view to a target projection?"*

This project introduces a novel, explainable approach to 3D inference, detailed in the paper ["A Solution for the Fundamental Problem of 3D Inference based on 2D Representations"](https://arxiv.org/abs/2211.04691).

---

## Table of Contents

- [The Vision: Dynamic 3D Inference](#the-vision-dynamic-3d-inference)
  - [Dynamic Gradient Flows](#dynamic-gradient-flows)
  - [Platonic-representation based 3D inference](#platonic-representation-based-3d-inference)
  - [Countable Vision](#countable-vision)
- [Scientific Foundation](#scientific-foundation)
- [Applications & Demos](#applications--demos)
- [Setup & Installation](#setup--installation)
- [Quick Start](#quick-start)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [Logo and Attribution](#logo-and-attribution)
- [License](#license)

## The Vision: Dynamic 3D Inference

This project is built upon three main pillars:

### Dynamic Gradient Flows
This concept rethinks backpropagation for differentiable rendering. Instead of optimizing an image-based loss which often "bubbles" a new point into existence, this method controls the gradient flow to physically **translate an existing point** to its corresponding target position on the screen. It treats the problem as a multi-objective optimization where each point is an objective, leading to a more stable and meaningful optimization process.

### Platonic-representation based 3D inference
The projection of an object onto an image is a 2D instance of that object's ideal "Platonic representation". This project performs 3D inference directly on these representations built during training. Unlike other approaches, our inverse rendering solution integrates the 3D inference process into the training loop.

Consequently, if the target view changes due to a shift in 3D parameters (e.g., rotation or translation), the process doesn't restart from scratch. Instead, it efficiently adapts by "moving" the object to its new configuration.

### Countable Vision
This concept will be revealed in the second phase of the project.

## Scientific Foundation

The core ideas are formally presented in our research paper. We introduce a generalization of the Blind PnP problem and provide a gradient-descent-based solution. The experiments are designed as illustrations of the theory rather than competitive benchmarks.

> 📄 **Read the full paper on arXiv: [https://arxiv.org/abs/2211.04691](https://arxiv.org/abs/2211.04691)**

## Applications & Demos

* **Camera Calibration:** Using a **[Perspective-n-Image](https://medium.com/@thienan092/from-pni-camera-calibration-to-monocular-3d-scene-reconstruction-part-i-what-is-c80879815e55) solver** approach. 
  - You can explore the implementation, which uses the **[Sky-dataset](https://github.com/thienannguyen-cv/Sky-dataset)**, in the [Jupyter Notebook](https://github.com/thienannguyen-cv/SMap/blob/main/applications/Camera%20Calibration/camera-calibration.ipynb).
  - For a quick look at the application, see the interactive demo below.

  [![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=github)](https://thienannguyen-cv.github.io/cab-viz/)

* **Depth Estimation:** Based on reconstructing **a single, Platonic 3D model** that accounts for all its projections.
  - You can explore the implementation, which uses the **[12Lamp-dataset](https://github.com/thienannguyen-cv/Lamp-dataset)**, in the [Jupyter Notebook](https://github.com/thienannguyen-cv/SMap/blob/main/applications/Depth%20Estimation/depth-estimation.ipynb).
  - For a quick look at the application, see the interactive demo below.

  [![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=github)](https://thienannguyen-cv.github.io/dep-viz/)

## Setup & Installation

### Setup environment with Conda
Follow these steps to set up a complete development environment for SMap. This is the recommended approach for both using and contributing to the library.

#### 1. Prerequisites

* **Conda:** Ensure you have Anaconda or Miniconda installed.
* **(GPU) NVIDIA Driver:** If you want to use a GPU (highly recommended), make sure you have the latest NVIDIA driver that supports **CUDA 11.8 or newer**.
  * Visit the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page to update.
  * After installing and rebooting, open a Command Prompt and run `nvidia-smi` to check the supported CUDA version.

#### 2. Create the Conda Environment

1.  **Use the `environment.yml` file:**
    This project includes an `environment.yml` file with a reliable configuration to ensure PyTorch is installed correctly with CUDA support.

2.  **Create the environment:**
    Open **Anaconda Prompt** (not Git Bash or a standard cmd), navigate to the project's root directory, and run the following command:

    ```bash
    conda env create -f environment.yml
    ```
    > **Note:** If an environment with the same name (`smap-env`) already exists, it's best to remove it first with `conda env remove -n smap-env` before running the `create` command. This ensures a clean installation.

3.  **Activate the environment:**
    Once the installation is complete, activate the new environment:
    ```bash
    conda activate smap-env
    ```
    You will see the environment name `(smap-env)` at the beginning of your command prompt line.

#### 3. Verify the Installation

To ensure PyTorch was installed correctly with GPU support, you can run the following Python script.

```python
import torch

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available
print(f"Is CUDA available: {torch.cuda.is_available()}")

# If CUDA is available, display details
if torch.cuda.is_available():
    print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is running on CPU. Please check your NVIDIA driver and CUDA installation.")
```

Run this script from your Anaconda Prompt (with the `smap-env` environment activated). The expected output for a successful GPU setup is:
```
PyTorch version: 2.0.1
Is CUDA available: True
CUDA version PyTorch was compiled with: 11.8
...
```

### Install from PyPI
The recommended way to install `smap-torch` is from PyPI:
```bash
pip install smap-torch
```

### Development Installation
To set up SMap for development:
```bash
# 1. Clone the repository
git clone https://github.com/thienannguyen-cv/SMap.git
cd SMap

# 2. (Optional) Create and activate a virtual environment
# On Linux or macOS:
python -m venv venv && source venv/bin/activate
# On Windows:
python -m venv venv && venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip setuptools
pip install -r requirements.txt

# 4. Install SMap in editable mode
pip install -e .
```

## Quick Start

Here is a simple example of how to use the library. After installation, you can import and use the `SMap` classes in your Python project:

```python
import numpy
import torch
from smap import SMap, SMap3x3

# Assume you have an input tensor (e.g., a binary mask)
# Shape: (N, C, H, W)
N, C, H, W = [1, 4, 128, 256]
input_tensor = torch.randn(N, C, H, W)

# Initialize SMap
# Define an affine transformation for the camera
camera = numpy.array([[2304.5479, 0,  1686.2379], 
                      [0, 2305.8757, -0.0151],
                      [0, 0, 1.]], dtype=numpy.float32)

smap_model = SMap(H, W, camera, "cpu")

# Pass the tensor through the model to get the mapping result
output = smap_model(input_tensor)

# The output contains the result of the spatial mapping
print("Output shape:", output.shape)
```

## Roadmap

The next release will focus on:
* **Depth Estimation:** Implementing depth estimation for single-shape objects.

For more detailed plans, you can check our [Trello Board](https://trello.com/invite/b/66d545d4e065eebded9a9c8f/ATTI56f6dabcfab65e388e9fa66b42e77f6bE3EB9A69/smap-project-management).

## Contributing

Contributions are welcome! If you have suggestions or want to contribute code, please open an issue to discuss your ideas or submit a pull request.

## Citation

If you use SMap or its underlying concepts in your research, please cite our paper:
```bibtex
@article{nguyen2022solution,
  title={A Solution for a Fundamental Problem of 3D Inference based on 2D Representations},
  author={Thien An L. Nguyen},
  journal={arXiv preprint arXiv:2211.04691},
  year={2022}
}
```

## Logo and Attribution

The SMap logo (`logo.png`) was created for the SMap project by Thien An L.
Nguyen. Copyright 2026 Thien An L. Nguyen. It may be used to refer to the
SMap project in a truthful and non-misleading way, but should not be used to
imply third-party endorsement, affiliation, or product identity without
permission.

See [NOTICE](NOTICE) for attribution details.

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for more details.

<hr>

[Go to Top](#table-of-contents)
