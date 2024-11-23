import argparse
import os
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=0.1)
    parser.add_argument("--num_decay_steps", type=float, default=10000)

    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--replay_memory_size", type=int, default=100000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=2000)

    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def compute_epsilon(total_steps, initial_epsilon, final_epsilon, num_decay_steps):
    # 남은 decay steps 계산 (최소 0)
    remaining_decay = max(num_decay_steps - total_steps, 0)

    # Linear decay 계산
    epsilon_decay = remaining_decay * \
        (initial_epsilon - final_epsilon) / num_decay_steps

    # 최종 epsilon = 최소값 + decay값
    current_epsilon = final_epsilon + epsilon_decay

    return current_epsilon


def epsilon_greedy_policy(predictions, num_actions, epsilon):
    if random() <= epsilon:  # 탐험: 무작위 행동 선택
        return randint(0, num_actions - 1)
    else:  # 활용: 최대 Q-value 행동 선택
        return torch.argmax(predictions).item()


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_episodes = 0
    total_steps = 0

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    writer = SummaryWriter(opt.log_path)

    model = DQN().to(device)
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    replay_memory = deque(maxlen=opt.replay_memory_size)

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    while num_episodes < opt.num_episodes:
        num_episodes += 1
        episode_reward = 0
        episode_cleared_lines = 0
        state = env.reset().to(device)
        done = False

        # Episode 실행
        while not done:
            total_steps += 1

            # Epsilon-greedy로 행동 선택
            epsilon = compute_epsilon(
                total_steps=total_steps,
                initial_epsilon=opt.initial_epsilon,
                final_epsilon=opt.final_epsilon,
                num_decay_steps=opt.num_decay_steps)

            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)

            predictions = model(next_states)[:, 0]
            index = epsilon_greedy_policy(
                predictions, len(next_steps), epsilon)
            next_state = next_states[index, :].to(device)
            action = next_actions[index]

            # 환경과 상호작용
            reward, done = env.step(action, render=False)

            # 경험 저장
            replay_memory.append([state, reward, next_state, done])
            if len(replay_memory) > opt.replay_memory_size:
                replay_memory.popleft()

            state = next_state

            # Training step (일정 주기로 실행)
            if total_steps % opt.train_freq == 0 and len(replay_memory) >= opt.batch_size:
                # Batch 샘플링
                batch = sample(replay_memory, opt.batch_size)
                state_batch, reward_batch, next_state_batch, done_batch = zip(
                    *batch)

                # Batch 데이터 준비
                state_batch = torch.stack(
                    tuple(state for state in state_batch)).to(device)
                reward_batch = torch.from_numpy(
                    np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
                next_state_batch = torch.stack(
                    tuple(state for state in next_state_batch)).to(device)

                # Q-learning update
                with torch.no_grad():
                    next_prediction_batch = target_model(next_state_batch)
                q_values = model(state_batch)

                y_batch = torch.cat(
                    tuple(reward if done else reward + opt.gamma * prediction
                          for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch))
                )[:, None]

                # Optimization
                loss = F.mse_loss(q_values, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Train/Loss', loss, total_steps)

            # Target network update (일정 주기로 실행)
            if total_steps % opt.target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        # Episode 종료 시 로깅
        writer.add_scalar('Episode/Reward', env.score, num_episodes)
        writer.add_scalar('Episode/Cleared lines',
                          env.cleared_lines, num_episodes)

        if num_episodes % opt.save_interval == 0:
            torch.save(
                model, f"{opt.saved_path}/tetris_episode_{num_episodes}")
        print(
            f"Episode {num_episodes}, Steps {total_steps}, Reward {env.score}, Cleared lines {env.cleared_lines}")

    torch.save(model, f"{opt.saved_path}/tetris_episode_{num_episodes}")


if __name__ == "__main__":
    opt = get_args()

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)

    train(opt)
