import torch
import torch.nn as nn
import numpy as np


def check(input):

    output = np.array(input) if type(input) != np.ndarray else input
    output = torch.from_numpy(output)
    return output


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class FullyConnectedNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc_dynamics_layers, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_dynamics_layers = fc_dynamics_layers

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.dynamics_network = mlp(self.state_dim + self.action_dim,
                                    self.fc_dynamics_layers,
                                    self.state_dim)

    def dynamic(self, state, action):
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        state_action_pair = torch.cat((state, action), dim=-1)
        state_predicted = self.dynamics_network(state_action_pair)
        return state_predicted
