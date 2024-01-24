# learning-based-mppi
This is a Python implementation of the learning-based model predictive path integral (MPPI) algorithm introduced in [Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989202). The algorithm is trained and evaluated over Gymnasium task Cart-Pole-v1. This repo:
* offers a Python implementation of learning-based MPPI algorithm，which is not provided in the paper.
* develops an flexible and configurable code framework that encompasses data collection, networking training and evaluation. The modularized code structure is designed for ease of use.
* presents two-versions of the MPPI implementation: a conventional version that accurately reflects the algorithm but involves longer computational time; a recommended version that utilizes GPU to accelerate rollout through parallel computation.
