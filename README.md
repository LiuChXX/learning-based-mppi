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
For other information about Gymnasium, please refer to [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## How to run
When your environemnt is ready, you can directly run with command:
``` Bash
python main.py
```
By default, this will cover three processes: **data collection**, **network training** and **evaluation** on the cart-pole swing up task. The trained model will be saved in the "results" folder. For a quick evaluation without training, please set `is_train=False` in the `main.py` and set the `model_load_path` with the path of the model to be evaluated.

For other parameters, please see in `config.py`.

## Evaluation Performance
Here we present GIFs below to show an evaluation result on Gymnasium task Cart-Pole-v1.
