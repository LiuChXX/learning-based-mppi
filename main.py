import gymnasium
import torch
import os
import sys

from mppi.config import get_config
from mppi.runner.cartpole_runner import CartpoleRunner


def parse_args(args, parser):
    parser.add_argument("--if_train", action='store_false', default=True)
    parser.add_argument("--use_gpu_planner", action='store_false', default=True)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    env = gymnasium.make("CartPole-v1", render_mode="human")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_load_path = os.path.join(all_args.model_save_path, "dynamic_network_example.pkl")

    config = {
        "all_args": all_args,
        "env": env,
        "device": device,
        "model_load_path": model_load_path,
    }

    runner = CartpoleRunner(config)
    if all_args.if_train:
        runner.train()
    else:
        runner.evaluate()


if __name__ == '__main__':
    main(sys.argv[1:])
