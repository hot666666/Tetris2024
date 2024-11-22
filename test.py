import argparse

import torch
import cv2

from src.tetris import Tetris

MDOEL_NAME = ""


def get_args():
    parser = argparse.ArgumentParser("""Tetris2024 게임 환경 DQN 테스트""")
    parser.add_argument("--width", type=int, default=10,
                        help="The common width for all images")
    parser.add_argument("--height", type=int, default=20,
                        help="The common height for all images")
    parser.add_argument("--block_size", type=int,
                        default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=120,
                        help="frames per second")
    parser.add_argument("--saved_path", type=str,
                        default=f"trained_models/{MDOEL_NAME}")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--record", type=bool, default=False)

    args = parser.parse_args()
    return args


def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    out = None
    if opt.record:
        out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                              (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))

    model = torch.load(opt.saved_path, map_location=device)
    model = torch.compile(model)
    model = model.to(device).eval()

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    while True:
        # 현재 블록이 현재 높이에서 놓일 수 있는 모든 위치에 대해 상태를 얻고, 그 상태에 대한 q-value를 예측하여 최대값을 가지는 행동을 선택
        # 모델의 입력에 사용되는 데이터는 (삭제줄 수, 빈곳 합, 인접열간차이 합, 높이 합) -> extract_board_features
        # env.step()의 사용되는 액션은 (x, num_rotations) 형태

        next_steps = env.get_next_states()

        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        # stack -> [batch_size, 1] -> [:, 0] -> [batch_size]
        predictions = model(next_states)[:, 0]

        index = torch.argmax(predictions).item()
        action = next_actions[index]

        _, done = env.step(action, render=True, video=out)

        if done:
            if out:
                out.release()
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
