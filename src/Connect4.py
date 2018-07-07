from FixedMCTS import FixedMCTS
from DynamicMCTS import DynamicMCTS
from GameState import GameState
from proto.state_pb2 import State

import json
import numpy as np

class BoardState(GameState):
    Players = {0: ' ', 1 : 'X', 2 : 'O'}
    Width = 7
    Height = 6
    InARow = 4
    Dirs = [(0,1),(1,1),(1,0),(1,-1)]
    BoardShape = [Height, Width]
    LegalMoves = Width

    def __init__(self):
        self.GameType = 'Connect4'
        self.Board = np.zeros((self.Height, self.Width, 2), dtype=np.uint8)
        self.Player = 1
        self.PreviousPlayer = None

    def Copy(self):
        copy = BoardState()
        copy.Player = self.Player
        copy.Board = np.copy(self.Board)
        return copy

    def LegalActions(self):
        actions = np.zeros(self.Width)
        for j in range(self.Width):
            if np.sum(self.Board[self.Height-1, j, :]) == 0:
                actions[j] = 1

        return actions

    def LegalActionShape(self):
        return [self.LegalMoves]

    def ApplyAction(self, action):
        if np.sum(self.Board[self.Height -1, action, :]) != 0:
            raise ValueError('Tried to make an illegal move.')

        top = -1
        for i in reversed(range(self.Height)):
            if np.sum(self.Board[i, action, :]) != 0:
                top = i
                break

        self.Board[top + 1, action, self.Player - 1] = 1
        self.PreviousPlayer = self.Player
        self.Player = 1 if self.Player == 2 else 2

    def AsInputArray(self):
        player = np.full((self.Height, self.Width), 1 if self.Player == 1 else -1)
        array = np.zeros((1, self.Height, self.Width, 3), dtype=np.uint8)
        array[0, :, :, 0:2] = self.Board
        array[0, :, :, 2] = player
        return array

    def Winner(self, prevAction=None):
        board = self._collapsed()

        if prevAction is not None:
            for i in reversed(range(self.Height)):
                if np.sum(self.Board[i, prevAction, :]) != 0:
                    break
            win = self._checkVictory(board, i, prevAction)
            if win is not None: 
                return win
        else:
            for i in range(self.Height):
                for j in range(self.Width):
                    if board[i,j] == 0:
                        continue
                    win = self._checkVictory(board, i, j)
                    if win is not None: 
                        return win

        if self._isOver(board):
            return 0
        return None

    def EvalToString(self, eval):
        return str(eval)
        
    def SerializeState(self, state, policy, evaluation):
        serialized = State()

        serialized.player = state.Player
        serialized.mctsEval = evaluation
        serialized.mctsPolicy = policy.tobytes()
        serialized.boardEncoding = state.Board.tobytes()
        serialized.boardDims = np.array([self.Width, self.Height, 3]).tobytes()
        return serialized.SerializeToString()

    def DeserializeState(self, serialState):
        state = State()
        state.ParseFromString(serialState)
        dims = np.frombuffer(state.boardDims, dtype=np.uint8)

        return {
            'player': state.player,
            'mctsEval': state.mctsEval,
            'mctsPolicy': np.frombuffer(state.mctsPolicy, dtype=np.float),
            'board': np.frombuffer(state.boardEncoding, dtype=np.uint8).reshape(dims)
        }

    def _isOver(self, board):
        for j in range(self.Width):
            if np.sum(self.Board[self.Height-1, j, :]) == 0:
                return False
        return True

    def _checkVictory(self, board, i, j):
        p = board[i,j]
        for dir in self.Dirs:
            inARow = 0
            r = 0
            while r*dir[0] + i < self.Height and r*dir[1] + j < self.Width and r*dir[1] + j >= 0 and board[r*dir[0] + i, r*dir[1] + j] == p:
                inARow += 1
                r += 1
            r = -1
            while r*dir[0] + i >= 0 and r*dir[1] + j < self.Width and r*dir[1] + j >= 0 and board[r*dir[0] + i, r*dir[1] + j] == p:
                inARow += 1
                r -= 1
            if inARow >= self.InARow:
                return p
        return None

    def _collapsed(self):
        array = np.zeros(self.Board.shape[:2])
        for p in BoardState.Players:
            array[self.Board[:, :, p - 1] == 1] = p
        return array

    def __str__(self):
        array = self._collapsed()
        s = ''
        for i in reversed(range(array.shape[0])):
            s += '[ '
            for j in range(array.shape[1]):
                s += ' {} '.format(BoardState.Players[array[i, j]])
                if j < array.shape[1] - 1:
                    s += '|'
            s += ']\n'

        return s

    def __eq__(self, other):
        if other.Player != self.Player:
            return False
        return (other.Board == self.Board).all()

    def __hash__(self):
        return "{0}{1}".format(self.Player,str(self)).__hash__()

if __name__ == '__main__':
    params = {'maxDepth' : 10, 'explorationRate' : 1, 'playLimit' : 1000}
    player = FixedMCTS(**params)
    BoardState.Width = 7
    BoardState.Height = 6
    BoardState.InARow = 4

    state = BoardState()
    while state.Winner() is None:
        print(state)
        print('To move: {}'.format(state.Player))
        state, v, p = player.FindMove(state)
        print('Value: {}'.format(v))
        print('Selection Probabilities: {}'.format(p))
        print('Child Values: {}'.format(player.Root.ChildWinRates()))
        print('Child Exploration Rates: {}'.format(player.Root.ChildPlays()))
        print()
        player.MoveRoot([state])
    print(state)
    print(state.Winner())
