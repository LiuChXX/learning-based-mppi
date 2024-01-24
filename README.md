# learning-based-mppi
This repo presents a hand-made Python implementation of the learning-based model predictive path integral (MPPI) algorithm introduced in [Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989202) over [Gymnasium task Cart-Pole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/). A dynamic network is developed to approximate the system dynamics, and is incorporated into the MPPI algorithm. **Two version of the MPPI implementation** are offered: 
* A conventional version that accurately reflects the algorithm but involves longer computational time.
* A GPU-accelerated version that accelerate rollout through parallel computation, which is recommended.
The flexible and easily configurable code framework facilitates easy setup and further improvement.


## Installation

This is an example installation on CUDA == 12.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n mppi python=3.8
conda activate mppi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### Gymnasium
``` Bash
# install gymnasium
pip install numpy gymnasium[classic-control]
```
For details, please refer to [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
