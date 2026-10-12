import numpy as np
import torch

class TicTacToeBoard:
    def __init__(self):
        self.board = [0] * 9  # 0 = empty, 1 = X (Player 1), -1 = O (Player 2)
        self.turn = 1         # 1 starts (X)

    @property
    def legal_moves(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def push(self, move):
        self.board[move] = self.turn
        self.turn *= -1

    def copy(self):
        new_board = TicTacToeBoard()
        new_board.board = self.board[:]
        new_board.turn = self.turn
        return new_board

    def _check_winner(self):
        # All 8 winning lines
        lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in lines:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return 0

    def is_game_over(self):
        return self._check_winner() != 0 or 0 not in self.board

    def result(self):
        w = self._check_winner()
        if w == 1: return '1-0'
        elif w == -1: return '0-1'
        else: return '1/2-1/2'


def encode_board(board):
    """
    Converts a Tic-Tac-Toe board into a 3x3x3 PyTorch tensor.
    Channel 0: X pieces
    Channel 1: O pieces
    Channel 2: Whose turn it is (1s for X, 0s for O)
    """
    state = np.zeros((3, 3, 3), dtype=np.float32)
    for i, val in enumerate(board.board):
        r, c = divmod(i, 3)
        if val == 1:
            state[0, r, c] = 1.0
        elif val == -1:
            state[1, r, c] = 1.0
            
    if board.turn == 1:
        state[2, :, :] = 1.0
        
    return torch.from_numpy(state).unsqueeze(0)