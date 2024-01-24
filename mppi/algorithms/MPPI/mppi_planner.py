import torch
import numpy as np
import math
import random


def state_cost(state, coef):
    """calculate the state dependent cost"""
    x = state[0]
    x_dot = state[1]
    theta = state[2]
    theta = 0.5 * math.pi - abs(theta)
    theta_dot = state[3]

    cost = coef[0] * x ** 2 + coef[1] * (math.cos(theta) + 1) ** 2 + coef[2] * x_dot ** 2 + coef[3] * theta_dot ** 2

    return cost


def terminal_cost(state):
    """calculate the terminal cost"""
    x = state[0]
    theta = state[2]
    if theta < 0.05 and abs(x) < 4:
        cost = 0
    else:
        cost = 1000
    return cost


class planner:
    def __init__(self, rollout_num, horizon, variance, lamda, coef, net):
        self.rollout_num = rollout_num
        self.horizon = horizon
        self.variance = variance
        self.lamda = lamda
        self.control_law = self.control_law_init()
        self.coef = coef
        self.net = net

    def control_law_init(self):
        control_law_horizon = np.zeros(self.horizon)
        for i in range(self.horizon):
            control_law_horizon[i] = random.random() * 2 - 1  # 产生[-1,1]中的随机数
        return control_law_horizon

    def plan(self, state_init):
        """forward planning"""
        noises_list = []
        rollout_cost_list = np.zeros(self.rollout_num)
        for i in range(self.rollout_num):
            current_state = state_init
            noises_per_rollout = np.random.normal(0, self.variance, self.horizon)
            noises_list.append(noises_per_rollout)
            rollout_cost = 0
            for t in range(self.horizon):
                control = self.control_law[t] + noises_per_rollout[t]
                control_int = 0 if control <= 0 else 1
                with torch.no_grad():
                    next_state = self.net.dynamic(state=current_state, action=[control_int]).cpu().numpy()
                rollout_cost += state_cost(next_state, self.coef) + self.lamda * self.control_law[t] * (1 / self.variance) * noises_per_rollout[t]
                current_state = next_state
            rollout_cost += terminal_cost(next_state)
            rollout_cost_list[i] = rollout_cost

        beta = min(rollout_cost_list) * np.ones(self.rollout_num)
        ita = sum(np.exp((-1 * 1 / self.lamda) * (rollout_cost_list - beta)))
        weights = (1 / ita) * np.exp((-1 * 1 / self.lamda) * (rollout_cost_list - beta))

        for t in range(self.horizon):
            for i in range(self.rollout_num):
                self.control_law[t] += weights[i] * noises_list[i][t]

        return 0 if self.control_law[0] <= 0 else 1

    def plan_gpu(self, state_init):
        """forward planning using gpu for rollout acceleration"""
        rollouts_cost = torch.zeros(self.rollout_num)

        # achieve parallel rollout through torch.tensor
        current_state = torch.tensor(state_init)
        current_state = current_state.unsqueeze(0).repeat(self.rollout_num, 1)
        all_control_law = torch.tensor(self.control_law)
        all_control_law = all_control_law.unsqueeze(0).repeat(self.rollout_num, 1)

        noises = torch.normal(0, self.variance, (self.rollout_num, self.horizon))
        all_control_law_noise = all_control_law + noises

        for t in range(self.horizon):
            control = all_control_law_noise[:, t].unsqueeze(-1)
            control = torch.where(control < 0, 0., 1.)
            with torch.no_grad():
                next_state = self.net.dynamic(state=current_state, action=control).cpu()

            rollouts_cost += self.coef[0] * torch.square(next_state[:, 0]) \
                             + self.coef[2] * torch.square(next_state[:, 1]) \
                             + self.coef[1] * (torch.square(torch.cos(0.5 * math.pi - torch.abs(next_state[:, 2])) + 1)) \
                             + self.coef[3] * torch.square(next_state[:, 3]) \
                             + self.lamda * all_control_law[:, t] * (1 / self.variance) * noises[:, t]
            current_state = next_state

        rollouts_cost += torch.where((torch.abs(next_state[:, 0])) < 4 & (next_state[:, 2] < 0.05), 0., 1000.)

        rollouts_cost = rollouts_cost.numpy()
        beta = min(rollouts_cost) * np.ones(self.rollout_num)
        ita = sum(np.exp((-1 * 1 / self.lamda) * (rollouts_cost - beta)))
        weights = (1 / ita) * np.exp((-1 * 1 / self.lamda) * (rollouts_cost - beta))

        for t in range(self.horizon):
            self.control_law[t] += (weights * noises.numpy()[:, t]).sum()

        return 0 if self.control_law[0] <= 0 else 1

    def update(self):
        self.control_law = np.delete(self.control_law, [0])
        self.control_law = np.append(self.control_law, [random.random() * 2 - 1])
