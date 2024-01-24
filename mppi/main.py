import gymnasium
import torch
import os

from config import get_config
from runner.cartpole_runner import CartpoleRunner


if __name__ == '__main__':
    parser = get_config()
    all_args = parser.parse_args()

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
    is_train = True
    if is_train:
        runner.train()
    else:
        runner.evaluate()

