from collections import deque

import numpy as np
from PIL import Image
import cv2
import torch

from src.randomizer import BagRandomizer, RandRandomizer


class TetrominoQueue:
    """테트로미노를 미리 정해진 크기만큼 랜덤하게 뽑아서 저장하는 클래스

        Args:
            randomizer (Randomizer): 테트로미노를 랜덤하게 뽑는 객체
            size (int): 미리 뽑아 저장할 테트로미노 크기
    """

    def __init__(self, randomizer=BagRandomizer(), size=1):
        self.size = size
        self.randomizer = randomizer
        self.queue = deque([], maxlen=size)

    def pop(self) -> int:
        next_piece = self.queue.popleft()
        self.queue.append(self.randomizer.get_random())
        return next_piece

    def peek(self) -> int:
        return self.queue[0]

    def reset(self):
        self.queue.clear()
        for _ in range(self.size):
            self.queue.append(self.randomizer.get_random())


class TetrisRenderer:
    piece_colors = [
        (0, 0, 0),
        (127, 219, 255),
        (255, 210, 125),
        (192, 132, 151),
        (168, 230, 207),
        (255, 111, 97),
        (90, 125, 154),
        (255, 160, 122)
    ]

    text_color = (255, 255, 255)
    header_color = (169, 169, 169)
    font = cv2.FONT_HERSHEY_DUPLEX

    font_scale = 1.0
    thickness = 1

    def __init__(self, height=20, width=10, block_size=30):
        self.height = height
        self.width = width
        self.block_size = block_size

        # 헤더 보드
        self.extra_board = np.ones((self.height * int(self.block_size / 4), self.width * self.block_size, 3),
                                   dtype=np.uint8) * np.array(self.header_color, dtype=np.uint8)

    # TODO: 랜더링 최적화
    def render(self, game_state, video=None):
        board_state, score, next_piece = game_state

        # 메인 보드 랜더링
        board_img = [self.piece_colors[p]
                     for row in board_state for p in row]
        board_img = np.array(board_img).reshape(
            (self.height, self.width, 3)).astype(np.uint8)
        board_img = board_img[..., ::-1]   # RGB -> BGR 변환
        board_img = Image.fromarray(board_img, "RGB")

        board_img = board_img.resize((self.width * self.block_size,
                                      self.height * self.block_size), 0)
        board_img = np.array(board_img)
        board_img[[i * self.block_size for i in range(self.height)], :, :] = 0
        board_img[:, [i * self.block_size for i in range(self.width)], :] = 0

        board_img = np.concatenate((self.extra_board, board_img), axis=0)

        # 다음 테트로미노(next_piece) 4x4로 패딩
        padded_piece = np.zeros((4, 4), dtype=int)
        piece_h, piece_w = len(next_piece), len(next_piece[0])
        padded_piece[:piece_h, :piece_w] = next_piece

        next_piece_img = [self.piece_colors[p]
                          for row in padded_piece for p in row]
        next_piece_img = np.array(next_piece_img).reshape(
            (4, 4, 3)).astype(np.uint8)
        next_piece_img = next_piece_img[..., ::-1]  # RGB -> BGR 변환
        next_piece_img = Image.fromarray(next_piece_img, "RGB")
        next_piece_img = next_piece_img.resize(
            (4 * self.block_size // 2, 4 * self.block_size // 2), resample=Image.NEAREST)

        next_piece_img = np.array(next_piece_img)
        next_piece_img[[i * self.block_size //
                        2 for i in range(4)], :, :] = 0
        next_piece_img[:, [i * self.block_size //
                           2 for i in range(4)], :] = 0

        # "Score" 텍스트 위치
        score_text = "Score"
        (score_width, _), _ = cv2.getTextSize(
            score_text, self.font, self.font_scale, self.thickness)
        score_x = int((self.width * self.block_size / 4) -
                      (score_width / 2))
        cv2.putText(board_img, score_text, (score_x, self.block_size),
                    fontFace=self.font, fontScale=self.font_scale, color=self.text_color)

        # "Score" 값 위치
        score_value = str(score)
        (value_width, _), _ = cv2.getTextSize(
            score_value, self.font, self.font_scale, self.thickness)
        value_x = int((self.width * self.block_size / 4) -
                      (value_width / 2))
        cv2.putText(board_img, score_value, (value_x, 3 * self.block_size),
                    fontFace=self.font, fontScale=self.font_scale, color=self.text_color)

        # "Next" 텍스트 위치
        next_text = "Next"
        (next_width, _), _ = cv2.getTextSize(
            next_text, self.font, self.font_scale, self.thickness)
        next_x = int((self.width * self.block_size * 3 / 4) -
                     (next_width / 2))
        cv2.putText(board_img, next_text, (next_x, self.block_size),
                    fontFace=self.font, fontScale=self.font_scale, color=self.text_color)

        # "Next" 아래에 next_piece 이미지 추가
        next_piece_x = int((self.width * self.block_size *
                           3 / 4) - next_piece_img.shape[1] // 2)
        next_piece_y = 2 * self.block_size
        board_img[next_piece_y:next_piece_y + next_piece_img.shape[0],
                  next_piece_x:next_piece_x + next_piece_img.shape[1]] = next_piece_img

        if video:
            video.write(board_img)

        cv2.imshow("Tetris2024 with DQN", board_img)
        cv2.waitKey(1)


class Tetris:
    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    rotation_mapper = [1, 4, 2, 2, 2, 4, 4]

    def __init__(self, height=20, width=10, block_size=30, randomizer=BagRandomizer()):
        self.height = height
        self.width = width
        self.queue = TetrominoQueue(randomizer=randomizer)
        self.renderer = TetrisRenderer(height, width, block_size)
        self.reset()

        """게임 상태관련 변수
        - board : 현재 보드 상태
        - score : 현재 점수
        - cleared_lines : 지워진 줄 수
        - gameover : 게임 종료 여부
        - current_pos : 현재 블록 위치(x, y)
        - piece : 현재 블록(2차원 배열)
        - idx: 현재 종류 블록 인덱스
        """

    def reset(self):
        """게임 상태 초기화 후, 초기 상태 특징을 반환하는 메서드"""
        self.board = [[0] * self.width for _ in range(self.height)]

        self.score = 0
        self.cleared_lines = 0
        self.queue.reset()

        self.idx = self.queue.pop()
        self.piece = [r[:] for r in self.pieces[self.idx]]
        self.current_pos = {"x": self.width //
                            2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False

        return self.extract_board_features(self.board)

    def get_next_states(self):
        """현재 상태에서 가능한 모든 열(x)에서 가능한 모든 회전(num_rotations)에 대한 다음 상태를 반환하는 메서드
            states[(x, num_rotations)] -> extract_board_features"""

        states = {}

        curr_piece = [r[:] for r in self.piece]
        piece_id = self.idx
        num_rotations = self.rotation_mapper[piece_id]

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [r[:] for r in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)

                board = self.get_board_with_piece(piece, pos)

                states[(x, i)] = self.extract_board_features(board)
            curr_piece = self.get_rotated_piece(curr_piece)
        return states

    def step(self, action, render=True, video=None):
        """action(x, num_rotations)을 받아서 게임을 진행하고, 보상과 게임 종료 여부를 반환하는 메서드
            보드 상단(x, 0)에서 시작하는 블록이, 보드에 닿을 때까지 떨어지는 것을 구현"""

        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}

        for _ in range(num_rotations):
            self.piece = self.get_rotated_piece(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True
            self.score -= 5

        self.board = self.get_board_with_piece(self.piece, self.current_pos)
        lines_cleared, self.board = self.clear_full_rows(
            self.board)

        score = self.get_reward(lines_cleared)
        self.score += score
        self.cleared_lines += lines_cleared

        if not self.gameover:
            self.spawn_next_piece()

        return score, self.gameover

##################################################

    def check_collision(self, piece, pos):
        """현재 보드 상태에서, piece가 pos에 추가될 때 충돌이 발생하는지 여부를 반환하는 메서드"""
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def get_board_with_piece(self, piece, pos):
        """현재 보드의 복사본을 만들어서, piece를 pos에 추가한 보드를 반환하는 메서드"""

        board = [r[:] for r in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def truncate(self, piece, pos):
        # 현재 보드에 대해선 수정하는 작업 없이, 게임종료 여부 반환
        # 이때 piece가 보드 밖으로 나가는 경우, in-place 연산으로 piece를 잘라서 보드 안에 들어오게 함

        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def spawn_next_piece(self):
        """다음 테트로미노를 뽑아서 현재 테트로미노로 설정하는 메서드"""

        self.idx = self.queue.pop()
        self.piece = [r[:] for r in self.pieces[self.idx]]
        self.current_pos = {
            "x": self.width // 2 - len(self.piece[0]) // 2,  "y": 0
        }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def get_rotated_piece(self, piece):
        """현재 테트로미노를 시계방향으로 90도 회전한 결과를 반환하는 메서드"""

        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_reward(self, lines_cleared):
        """지워진 줄 수에 대한 보상을 반환하는 메서드"""
        return 1 + (lines_cleared ** 2) * self.width

##################################################

    def extract_board_features(self, board):
        """현재 보드 상태에 대한 특징(지워진 줄, 구멍, 인접열 차이 합, 높이 합)을 반환하는 메서드"""

        lines_cleared, board = self.clear_full_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def clear_full_rows(self, board):
        """보드에서 꽉 찬 줄을 지우고, 지워진 줄 수와 보드를 반환하는 메서드"""
        # in-place로 가득 찬 줄을 지우고, 지워진 줄 수와 보드를 반환

        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self._remove_rows(board, to_delete)
        return len(to_delete), board

    def get_holes(self, board):
        """보드에서 구멍 수를 반환하는 메서드"""
        # 위에서부터 블록있는 곳까지 내려가고, 이후부터 빈칸을 세는 방식으로 구멍 수를 반환

        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        """보드에서 인접 열간 높이 차이와 각 열의 높이 합을 반환하는 메서드"""
        # 인접 열간 높이 차이인 diffs의 합 total_bumpiness, 각 열의 높이 hight의 합 hights를 반환

        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(
            mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def _remove_rows(self, board, indices):
        # 보드에서 indices에 해당하는 행을 in-place 삭제하고, 위에 빈 행을 추가하는 메서드
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

##################################################

    def get_render_state(self):
        """랜더링을 위한 현재 게임 상태(보드, 점수, 다음 블록) 정보를 반환하는 메서드
            보드는 현재 블록(piece)를 추가한 보드로 반환, board += piece[current_pos]"""

        next_idx = self.queue.peek()
        next_piece = self.pieces[next_idx]

        if self.gameover:
            return self.board, self.score, next_piece

        board = [r[:] for r in self.board]
        curr_x, curr_y = self.current_pos["x"], self.current_pos["y"]

        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[curr_y+y][curr_x+x] += self.piece[y][x]

        return board, self.score, next_piece

    def render(self, video=None):
        """게임 상태를 렌더링하는 메서드"""
        self.renderer.render(self.get_render_state(), video)
