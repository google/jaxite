<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <img alt="FEATHER" src="figure_drawer/Morph_logo.png" width=15%>
</p>


<h3 align="center">
TPU-accelerated, free, immediate, fast, and cheap HE serving for everyone
</h3>
<p align="center">
| <a href="https://arxiv.org/abs/2604.17808">paper</a> |
<a href="https://github.com/EfficientPPML/MORPH">code</a> |
<a href="https://efficientppml.github.io/CROSS_Tutorial/">tutorial</a> |
</p>

🔥 We have delivered a tutorial at ASPLOS'26 to help you get started with MORPH. Please visit [CPA_tutorial](https://efficientppml.github.io/CROSS_Tutorial/) to learn more.
For questions, please drop an email to our community [email](cpacommunity@googlegroups.com).

---

# MORPH: Enable AI Accelerator for Zero Knowledge Proof
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)  

# 1. What is MORPH?
MORPH is the first project to enable AI Accelerator, such as Google TPUs, to accelerate Zero Knowledge Proof Primitives (Multi-scalar Multiplication and Number Theory Transformation) and achieves the State-of-the-art (SotA) throughput and energy efficiency (performance per watt). Together with [CROSS](https://github.com/EfficientPPML/CROSS), they enable AI ASICs to be SotA throughput machine for cryptography primitive with wide-range precision.

<img src="./figure_drawer/cross_morph.png" width="800">

It features 
- MXU Lazy Modular Reduction: bringing quadratic high-precision modular reduction down to linear operation.

<img src="./figure_drawer/morph_arithmetic_contribution.png" width="800">

- dataflow optimization for MSM and NTT. Details in the [paper](https://arxiv.org/abs/2604.17808).

This branch (`asplos`) contains demo scripts for profiling and comparing the two core workloads.

## Project Structure

```
├── finite_field_context.py           # Finite field arithmetic (MORPH & CROSS backends)
├── elliptic_curve_context.py         # Elliptic curve point arithmetic
├── multiscalar_multiplication_context.py    # Multi-scalar multiplication (MSM)
├── number_theory_transform_context.py       # Number Theoretic Transform (NTT)
├── utils.py                          # JAX kernel utilities, number theory helpers
├── profiler.py                       # Trace parsing and kernel profiling
├── configurations.toml               # Curve parameters (BLS12-377)
├── c_kernels/                        # Custom C kernels for TPU acceleration
├── deployments/                      # Serialized compiled JAX kernels
```
All functions have `_test.py` and `_perf_test.py` for correctness and performance testing.


## Key Concepts

| Concept | Description |
|---------|-------------|
| **DRNS (Double RNS)** | Residue Number System representation enabling efficient large-integer modular arithmetic on TPU |
| **MORPH** | Alternative modular multiplication backend using chunk-based representation |
| **MSM** | Multi-scalar multiplication — computing $\sum_i s_i \cdot P_i$ over elliptic curve points |
| **Bucket Accumulation** | MSM decomposition strategy: scalars are sliced into windows, points accumulated into buckets per window |
| **Compiled Kernels** | Pre-compiled JAX/C kernels stored in `deployments/` for fast TPU execution |
| **Sharding** | Distribution of computation aMORPH TPU cores |


# 2. Environment Setup

Inside TPU VM, please do following setup to configure the environment.

Step 1: install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Step 2: create environment and install required packages
```
source ~/.bashrc
conda create --name jaxite python=3.13
conda activate jaxite
pip install -U "jax[tpu]"
pip install xprof
pip install absl-py
pip install toml
pip install gdown
pip install pandas
pip install gmpy2
```

Step 3: Install the C++ toolchain for the MSM C kernel.

The MSM path uses a CPU C kernel (`c_kernels/distribution.cpp`) that is
compiled on the first import of `multiscalar_multiplication_context`. You need
a host `g++` with OpenMP, and the conda env's bundled `libstdc++` must be
recent enough to satisfy the symbols emitted by that compiler. On modern Ubuntu
(g++ 13+) this means `GLIBCXX_3.4.32`, which the conda default
`libstdcxx-ng 11.2.0` does **not** ship — so install a newer one from
`conda-forge`:
```
sudo apt-get install -y g++          # skip if already installed
conda install -n jaxite -c conda-forge 'libstdcxx-ng>=13' 'libgcc-ng>=13'
```
If you see `OSError: ... libstdc++.so.6: version 'GLIBCXX_3.4.XX' not found`
when importing `multiscalar_multiplication_context`, the conda libstdc++ is
older than your system `g++` — re-run the `conda install` line above (or
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` for a one-shot
workaround).

Step 4: Download the reference data
```
mkdir -p data && gdown 1aJhANlS8hWrjSt9j0nBKoFRBoZh0W1aa -O data/data.tar.gz && cd data && tar -xvf data.tar.gz
```

Step 5 (optional): Pre-build the MSM C kernel.

The C kernel is compiled automatically by `c_kernels/build.py` on the first
import of `multiscalar_multiplication_context`, so no separate build step is
required. To pre-build (or force a rebuild) ahead of time:
```
python -m c_kernels.build           # build if missing or stale
python -m c_kernels.build --force   # always rebuild
```
The compiler defaults to `g++` with `-std=c++17 -fopenmp -O2 -fPIC -shared`
plus `-I<jaxlib>/include`. Override via the `CXX` and `CXXFLAGS` env vars
(e.g. point `CXX` at conda's `gxx_linux-64` to keep the build inside the env).

# 3. TPU Setup
The code is optimized for TPU execution, but it also runs on NVIDIA GPU and CPU for functional preview (not optimized for these devices).

- Step 1: Create a Google Project [tutorial](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project).

Obtain the name of the project as <google_project_name> and **Google Project ID** from the created project.

- Step 2: Apply for the Tree-tier TPU trail for 30 days[TRC](https://sites.research.google/trc/about/)

Once submitted the request, an email will be shot to you within one day, where there is a link to fill in a survey with your **Google project ID**.

- Step 3: Launch TPU VM.
You could do it over GUI or gcloud cli (in your local machine) to create a TPU VM. I give the gcloud cli as it works for all generations (>=v4) of TPUs.

For TPUv6e,
```bash
gcloud config set project <google_project_name>
gcloud config set compute/zone us-east1-d
gcloud alpha compute tpus queued-resources create <google_project_name> --node-id=<your_favoriate_node_name> \
    --zone=us-east1-d \
    --accelerator-type=v6e-1  \
    --runtime-version=v2-alpha-tpuv6e \
    --provisioning-model=spot
```

Note that TPUv5e and TPUv6e could only work with provisioning-model as spot, because they are popular resources, and Google cloud can preempt it if there are tasks with higher priority requiring these resources. But you could get a long-term active TPUv4 VM as it's less demanding by other tasks.

- Step 4: Setup Remote SSH (VSCode or Cursor) to TPU VM
Once the requested TPU vm is up and running as shown in Google console, you could use gcloud to forward the SSH port of the remote machine to a port of local machine and setup VSCode remote ssh.

You need to first setup local ssh key to Google's compute engine, following [link](https://cloud.google.com/compute/docs/connect/create-ssh-keys#gcloud). After your follow the instructions on the page, the ssh key will be dumped here `<path_to_local_user>/.ssh/google_compute_engine`.


```bash
gcloud compute tpus tpu-vm ssh <gcloud_user_name>@<your_favoriate_node_name> -- -L 9009:localhost:22
```
Where 9009 is the port of local machine, while 22 is the SSH port of the TPU vm.

After you set it up, you could configure VSCode to use the remote SSH package [link](https://code.visualstudio.com/docs/remote/ssh) to remotely access into TPUvm.
```bash
Host tpu-vm
    User <gcloud_user_name>
    HostName localhost
    Port 9009
    IdentityFile  <path_to_local_user>/.ssh/google_compute_engine
```

After this, you should follow the steps on [link](https://code.visualstudio.com/docs/remote/ssh) to log into TPU VM.

# 4. Ready to Play?

Run functional correctness tests for both NTT and MSM:
```
python3 number_theory_transform_test.py
python3 multiscalar_multiplication_test.py
```

Run performance tests for both NTT and MSM:

```
python3 number_theory_transform_perf_test.py
python3 multiscalar_multiplication_perf_test.py
```

Notes:
- The first MSM test run auto-compiles `c_kernels/distribution.cpp` into
  `c_kernels/distribution.so` (a few seconds). Subsequent runs reuse the cached
  `.so` and rebuild only when the source is newer.
- The first run of each test also JIT-compiles JAX kernels; expect a longer
  first iteration that is then cached under `deployments/`.
- Performance tests assume the reference data from Step 4 is present under
  `./data/`.

# 5. Call for Actions
Our mission is to build an open-sourced SoTA library for the community.
- If you find this repository helpful, please consider giving it a star :)
- For any questions, please feel free to open an issue.
- For any suggestions or new features, please feel free to open a pull request.

# Contact
- Jianming Tong, Georgia Institute of Technology / Google, jianming.tong@gatech.edu/jianmingt@google.com
- Jingtian Dang, Georgia Institute of Technology, dangjingtian@gatech.edu
- Tushar Krishna, Georgia Institute of Technology, tushar@ece.gatech.edu


# Citation

```
@inproceedings{tong2025MORPH,
author = {Jianming Tong and Jingtian Dang and Simon Langowski and Tianhao Huang and Asra Ali and Jeremy Kun and Srini Devadas and Tushar Krishna},
title = {MORPH: Enabling AI ASICs for Zero Knowledge Proof},
year = {2026},
publisher = {IEEE Press},
booktitle = {Proceedings of the 63nd Annual ACM/IEEE Design Automation Conference},
location = {Los Angeles, California, United States},
series = {DAC '26}
}
```

Enjoy! :D
