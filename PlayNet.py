import os
import sys
import yaml
import numpy as np
sys.path.insert(0, './src/')

from Blackbird import BlackBird
from TicTacToe import BoardState
from GameState import GameState


if __name__ == '__main__':
    assert os.path.isfile('parameters.yaml'), 'Copy the parameters_template.yaml file into parameters.yaml to test runs.'
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())

    BlackbirdInstance = BlackBird(tfLog=False, loadOld=True, **parameters)

    state = BoardState()
    assert isinstance(state, GameState)
    winner = None
    aiToPlay = np.random.choice([True, False])
    print(state)
    while winner is None:
        if aiToPlay:
            nextState, _, _ = BlackbirdInstance.FindMove(state)
        else:
            move = int(input('Enter a move: '))
            state.ApplyAction(move)
            nextState = state
        aiToPlay = not aiToPlay
        BlackbirdInstance.MoveRoot([nextState])
        print(nextState)
        print()
        state = nextState
        winner = state.Winner()

    print('Winner : {}'.format(BoardState.Players[winner]))


