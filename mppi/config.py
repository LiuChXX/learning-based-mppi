import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(
        description='learning_based_mppi', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--buffer_capacity", type=int, default=10000, help="Number of capacity for replay buffer")

    parser.add_argument("--layer_num", type=int, default=1, help="Number of MLP layers in dynamic model")
    parser.add_argument("--layer_size", type=int, default=64, help="Number of layer size")

    parser.add_argument("--training_epochs", type=int, default=200, help="Number of epochs for network training")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of batch size for network training")
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate for network training")
    parser.add_argument("--lr_decay_rate", type=float, default=0.9, help="lr decay rate for network training")

    parser.add_argument("--mppi_rollout_num", type=int, default=50, help="Number of rollout for mppi planner")
    parser.add_argument("--mppi_horizon", type=int, default=5, help="Number of prediction horizon for mppi planner")
    parser.add_argument("--mppi_variance", type=int, default=1, help="variance for mppi planner")
    parser.add_argument("--lamda", type=float, default=1, help="lamda for mppi planner")

    parser.add_argument("--cart_position_coef", type=int, default=10,
                        help="coefficient parameter for cost calculation")
    parser.add_argument("--cart_velocity_coef", type=int, default=1,
                        help="coefficient parameter for cost calculation")
    parser.add_argument("--pole_angle_coef", type=int, default=500,
                        help="coefficient parameter for cost calculation")
    parser.add_argument("--pole_angle_velocity_coef", type=int, default=15,
                        help="coefficient parameter for cost calculation")

    parser.add_argument("--model_save_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                        help="set the path to save trained model")

    parser.add_argument("--evaluate_num", type=int, default=10, help="Number of episodes of a single evaluation")
    parser.add_argument("--evaluate_time_limit", type=int, default=500, help="time limit for each episode in evaluation")

    return parser
