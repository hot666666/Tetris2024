import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from src.dqn import DQN
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser("""Tetris2024 게임 환경 DQN 학습""")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=6000)
    parser.add_argument("--save_interval", type=int, default=300)
    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--target_network_frequency", type=int, default=2000)

    args = parser.parse_args()
    return args


def compute_epsilon(epoch, initial_epsilon, final_epsilon, num_decay_epochs):
    epsilon_decay = max(num_decay_epochs - epoch, 0) * \
        (initial_epsilon - final_epsilon) / num_decay_epochs

    return final_epsilon + epsilon_decay


def epsilon_greedy_policy(predictions, num_actions, epsilon):
    if random() <= epsilon:
        # 탐험: 무작위 행동 선택
        return randint(0, num_actions - 1)
    else:
        # 활용: 최대 Q-value 행동 선택
        return torch.argmax(predictions).item()


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    model = DQN().to(device)
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    replay_memory = deque(maxlen=opt.replay_memory_size)

    state = env.reset()
    state = state.to(device)

    epoch = 0
    while epoch < opt.num_epochs:
        epsilon = compute_epsilon(epoch=epoch,
                                  initial_epsilon=opt.initial_epsilon,
                                  final_epsilon=opt.final_epsilon,
                                  num_decay_epochs=opt.num_decay_epochs)

        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        predictions = model(next_states)[:, 0]

        index = epsilon_greedy_policy(predictions, len(next_steps), epsilon)
        next_state = next_states[index, :].to(device)
        action = next_actions[index]

        reward, done = env.step(action, render=False)

        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_cleared_lines = env.cleared_lines
            state = env.reset().to(device)
        else:
            state = next_state
            continue

        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.stack(
            tuple(state for state in state_batch)).to(device)
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
        next_state_batch = torch.stack(
            tuple(state for state in next_state_batch)).to(device)

        with torch.no_grad():
            next_prediction_batch = target_model(next_state_batch)
        q_values = model(state_batch)

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction
                  for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch))
        )[:, None]

        loss = F.mse_loss(q_values, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % opt.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(
                target_model.parameters(), model.parameters()
            ):
                target_network_param.data.copy_(
                    opt.tau * q_network_param.data +
                    (1.0 - opt.tau) * target_network_param.data
                )

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch}/{opt.num_epochs}, Score: {final_score}, : Cleared lines: {final_cleared_lines}")

        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Cleared lines',
                          final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, f"{opt.saved_path}/tetris_{epoch}")

    torch.save(model, f"{opt.saved_path}/tetris_{epoch}")


if __name__ == "__main__":
    opt = get_args()

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    train(opt)
