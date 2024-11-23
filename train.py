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
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=0.01)
    parser.add_argument("--num_decay_steps", type=float, default=5000)

    parser.add_argument("--num_episodes", type=int, default=40000)
    parser.add_argument("--replay_memory_size", type=int, default=300000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_update_freq", type=int, default=4000)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


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
        state = env.reset().to(device)
        done = False

        total_loss = 0
        episode_steps = 0

        # Episode 실행
        while not done:
            total_steps += 1
            episode_steps += 1

            # Epsilon 값 계산
            epsilon = max(opt.final_epsilon, opt.initial_epsilon - (
                opt.initial_epsilon - opt.final_epsilon) * (total_steps / opt.num_decay_steps))

            # Epsilon 값 로깅 (매 스텝마다)
            writer.add_scalar('Train/Epsilon', epsilon, total_steps)

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

            # 상태 업데이트
            state = next_state

            # Training step
            if len(replay_memory) < max(opt.batch_size, opt.replay_memory_size // 10):
                continue

            # train_freq만큼 학습
            if total_steps % opt.train_freq != 0:
                continue

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

            # Q-value 예측
            q_values = model(state_batch)

            # Target Q-value 계산
            with torch.no_grad():
                next_prediction_batch = target_model(next_state_batch)

            opt_q_values = reward_batch + opt.gamma * next_prediction_batch.max(
                1, keepdim=True)[0] * (1 - torch.tensor(done_batch).float().unsqueeze(1).to(device))

            # Optimization
            loss = F.mse_loss(q_values, opt_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 학습 단계별 손실 로깅 (매 스텝)
            writer.add_scalar('Train/StepLoss', loss.item(), total_steps)

            # Target network update (일정 주기로 실행)
            if total_steps % opt.target_update_freq == 0:
                for target_param, param in zip(target_model.parameters(), model.parameters()):
                    target_param.data.copy_(
                        opt.tau * param.data + (1 - opt.tau) * target_param.data)

        # Episode 종료 시 로깅
        if num_episodes < 2000 and num_episodes % opt.save_interval == 0:
            torch.save(
                model, f"{opt.saved_path}/tetris_episode_{num_episodes}")

        # 에피소드 로그 출력
        avg_loss = total_loss / episode_steps if total_loss > 0 else 0
        if avg_loss > 0:
            print(
                f"Episode {num_episodes}, Total Steps {total_steps}, Episode Steps {episode_steps}, Avg loss {avg_loss},  Reward {env.score}, Cleared lines {env.cleared_lines}"
            )

            # Episode 종료 시 로깅
            writer.add_scalar('Episode/AverageLoss', avg_loss, num_episodes)
            writer.add_scalar('Episode/Reward', env.score, num_episodes)
            writer.add_scalar('Episode/Cleared lines',
                              env.cleared_lines, num_episodes)
        else:
            print(
                f"Episode {num_episodes}, Total Steps {total_steps}, Episode Steps {episode_steps}, Reward {env.score}, Cleared Lines {env.cleared_lines}")

    torch.save(model, f"{opt.saved_path}/tetris_episode_{num_episodes}")


if __name__ == "__main__":
    opt = get_args()

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)

    train(opt)
