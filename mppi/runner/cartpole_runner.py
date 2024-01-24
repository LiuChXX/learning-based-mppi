import torch
import numpy as np
import os
import time

from mppi.algorithms.MPPI import mppi_planner
from mppi.algorithms.MPPI import model
from mppi.algorithms.MPPI import buffer
from mppi.algorithms.MPPI import trainer


class CartpoleRunner:
    def __init__(self, config):
        self.all_args = config['all_args']
        self.env = config['env']
        self.device = config['device']

        # parameters
        self.buffer_capacity = self.all_args.buffer_capacity

        self.layer_num = self.all_args.layer_num
        self.layer_size = self.all_args.layer_size

        self.training_epochs = self.all_args.training_epochs
        self.batch_size = self.all_args.batch_size
        self.lr_init = self.all_args.lr_init
        self.lr_decay_rate = self.all_args.lr_decay_rate

        self.mppi_rollout_num = self.all_args.mppi_rollout_num
        self.mppi_horizon = self.all_args.mppi_horizon
        self.mppi_variance = self.all_args.mppi_variance
        self.lamda = self.all_args.lamda

        self.cart_position_coef = self.all_args.cart_position_coef
        self.cart_velocity_coef = self.all_args.cart_velocity_coef
        self.pole_angle_coef = self.all_args.pole_angle_coef
        self.pole_angle_velocity_coef = self.all_args.pole_angle_velocity_coef

        self.if_use_gpu_rollout = self.all_args.use_gpu_accelerate_rollout

        self.model_save_path = self.all_args.model_save_path
        self.model_load_path = config['model_load_path']

        self.evaluate_num = self.all_args.evaluate_num
        self.evaluate_time_limit = self.all_args.evaluate_time_limit

        self.net = model.FullyConnectedNetwork(state_dim=4,
                                               action_dim=1,
                                               fc_dynamics_layers=self.layer_num*[self.layer_size],
                                               device=self.device).to(self.device)

    def train(self):
        """dynamic network training"""
        replay_buffer = buffer.ReplayBuffer(self.buffer_capacity)
        my_trainer = trainer.Trainer(net=self.net,
                                     epochs=self.training_epochs,
                                     batch_size=self.batch_size,
                                     lr_init=self.lr_init,
                                     lr_decay_rate=self.lr_decay_rate,
                                     replaybuffer=replay_buffer,
                                     device=self.device)
        """sample the transition data in the environment using for training the dynamic network"""
        for episode in range(500):
            observation, info = self.env.reset()  # reset the environment
            print("sampling in process, episode = {} in {}".format(episode+1, 500))
            for timestep in range(200):
                old_obs = observation
                action = self.env.action_space.sample()  # random actions
                observation_, reward, terminated, truncated, info = self.env.step(action)  # one-step interaction
                new_obs = observation_
                # print(state_cost(new_obs[0], new_obs[1], new_obs[2], new_obs[3]))

                replay_buffer.push([old_obs, [action], reward, new_obs, terminated])
                if terminated:
                    break
                observation = observation_

        """dynamic network training"""
        my_trainer.train()
        t = time.localtime()
        save_path = os.path.join(self.model_save_path, './dynamic_network_{}_{}_{}_{}_{}_{}.pkl'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
        torch.save(self.net.dynamics_network.state_dict(), save_path)

        """dynamic network testing"""
        self.net.dynamics_network.load_state_dict(torch.load(save_path))
        transitions = replay_buffer.buffer[0: 2]
        batch_obs, batch_actions, batch_reward, batch_next_obs, batch_done = zip(*transitions)

        # list to np.ndarray
        batch_obs = np.array(batch_obs)
        batch_actions = np.array(batch_actions)
        batch_next_obs = np.array(batch_next_obs)

        prediction = self.net.dynamic(state=batch_obs, action=batch_actions)
        batch_next_obs = torch.tensor(batch_next_obs, device=self.device, dtype=torch.float32)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(prediction, batch_next_obs)

        print(loss)

        """evaluate trained dynamic network with mppi planner"""
        self.model_load_path = save_path
        self.evaluate()

    def evaluate(self):
        """evaluation"""
        self.net.dynamics_network.load_state_dict(torch.load(self.model_load_path))

        for episode in range(self.evaluate_num):
            coef = [self.cart_position_coef, self.pole_angle_coef, self.cart_velocity_coef, self.pole_angle_velocity_coef]
            planner = mppi_planner.planner(self.mppi_rollout_num, self.mppi_horizon, self.mppi_variance, self.lamda, coef, self.net)
            state, info = self.env.reset()
            total_reward = 0

            print('Test_Episode {} in {}'.format(episode + 1, self.evaluate_num))
            for step in range(self.evaluate_time_limit):
                if self.if_use_gpu_rollout:
                    """use gpu for rollout acceleration"""
                    action = planner.plan_gpu(state)
                else:
                    action = planner.plan(state)
                planner.update()
                state, reward, terminated, truncated, info = self.env.step(action)

                total_reward += reward

                if terminated or truncated:
                    print("total_reward = {}".format(total_reward))
                    break

        self.env.close()
