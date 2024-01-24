# learning-based-mppi
This repo provides a hand-made Python implementation of the learning-based model predictive path integral (MPPI) algorithm introduced in [Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989202). A dynamic network is developed to approximate the system dynamics and is incorporated into the MPPI online planner. For the MPPI planner, **two implementation versions** are provided: 
* A conventional version that accurately reflects the algorithm but involves longer computational time (see function `plan` in `./algorithm/MPPI/mppi_planner.py`).
* A GPU-accelerated version that accelerate rollout through parallel computation, which is recommended (see function `plan_gpu` in `./algorithm/MPPI/mppi_planner.py`).

The flexible and easily configurable code framework facilitates easy setup and further improvement. The algorithm is trained and evaluated on a cart-pole swing up task [Gymnasium task Cart-Pole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/).


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
pip install gymnasium[classic-control]
```
For other information about Gymnasium, please refer to [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## How to run
When your environemnt is ready, you can directly run with command:
``` Bash
python main.py
```
By default, this will cover three processes: **data collection**, **network training** and **evaluation** on the cart-pole swing up task with **the GPU-accelerated MPPI planner**. The trained model will be saved in the "results" folder. For a quick evaluation without training, please set `is_train=False` in the `main.py` and set the `model_load_path` with the path of the model to be evaluated. For using MPPI planner of the conventional version, please set `--use_gpu_accelerate_rollout` in `config.py`.

For other parameters, please see in `config.py`.

## Evaluation Performance
Here we present GIF below to show an evaluation result on Gymnasium task Cart-Pole-v1.
|<img src="cartpole_gif.gif" align="middle" width="300" border="1"/>|
|:-------------------------: |
|Evaluation Result|  
