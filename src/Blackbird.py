import functools
import random
import yaml
import numpy as np
import os
import TicTacToe
import Connect4
import json
from Model import Model, TrainingExample

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)

defaultMCTS = {
    "maxDepth": 10,
    "explorationRate": 0.85,
    "timeLimit": None,
    "playLimit": 250,
    "temperature": {
        "exploration": 1,
        "exploitation": 0.1
    }
}

defaultTensorflow = {
    "GPUOptions": {
        "per_process_gpu_memory_fraction": 0.2
    }
}


# Basic

def _resolveGame(gameName):
    if gameName == 'TicTacToe':
        return TicTacToe.BoardState
    elif gameName == 'Connect4':
        return Connect4.BoardState
    raise ValueError('Unsupported game type: {}'.format(gameName))

def NewModel(name, networkConfig, mctsConfig=defaultMCTS, tensorflowConfig=defaultTensorflow):
    if name in Models():
        raise ValueError('A model with that name already exists. {}'.format(name))
    else:
        m = Model(name, 0, mctsConfig, networkConfig, tensorflowConfig)
        m.SaveMCTS()
        m.SaveTF()

def Models():
    if os.path.isdir(Model.ModelDirectory):
        return [folder for folder in os.listdir(Model.ModelDirectory)
                    if os.path.isdir(folder)]
    return []


def SetMCTSConfig(name, mctsConfig):
    modelDir = os.path.join(Model.ModelDirectory, name)
    if not os.path.isdir(modelDir):
        raise IOError('Model directory could not be found at {}'.format(
            os.path.abspath(modelDir)))
    fp = os.path.join(modelDir, 'mctsConfig.json')
    with open(fp, 'w') as fout:
        json.dump(mctsConfig, fp)


def GetMCTSConfig(name):
    fp = os.path.join(Model.ModelDirectory, name, 'mctsConfig.json')
    if not os.path.exists(fp):
        return {}

    with open(fp, 'r') as fin:
        return json.load(fin)

# Model Testing
def TestRandom(model, temp, numTests):
    return TestModels(model, RandomMCTS(), temp, numTests)


def TestPrevious(model, temp, numTests):
    """ Plays the current BlackBird instance against the previous version of
    BlackBird's neural network.

    Args:
        `model`: The Blackbird model to test
        `temp`: A float between 0 and 1 determining the exploitation
            temp for MCTS. Usually this should be close to 0.1 to ensure
            optimal move selection.
        `numTests`: An int determining the number of games to play.

    Returns:
        `wins`: The number of wins BlackBird had.
        `draws`: The number of draws BlackBird had.
        `losses`: The number of losses BlackBird had.
    """
    oldModel = model.LastVersion()

    results = TestModels(model, oldModel, temp, numTests)

    del oldModel
    return results


def TestGood(model, temp, numTests):
    """ Plays the current BlackBird instance against a standard MCTS player.

        Args:
            `model`: The Blackbird model to test
            `temp`: A float between 0 and 1 determining the exploitation
                temp for MCTS. Usually this should be close to 0.1 to ensure
                optimal move selection.
            `numTests`: An int determining the number of games to play.

        Returns:
            `wins`: The number of wins BlackBird had.
            `draws`: The number of draws BlackBird had.
            `losses`: The number of losses BlackBird had.
    """
    good = FixedMCTS(maxDepth=10, explorationRate=0.85, timeLimit=1)
    return TestModels(model, good, temp, numTests)


def TestModels(model1, model2, temp, numTests):
    wins = draws = losses = 0
    for _ in range(numTests):
        model1ToMove = random.choice([True, False])
        model1Player = 1 if model1ToMove else 2
        winner = None
        model1.DropRoot()
        model2.DropRoot()
        state = model1.Game()

        while winner is None:
            if model1ToMove:
                (nextState, *_) = model1.FindMove(state, temp)
            else:
                (nextState, *_) = model2.FindMove(state, temp)
            state = nextState
            model1.MoveRoot(state)
            model2.MoveRoot(state)

            model1ToMove = not model1ToMove
            winner = state.Winner()

        if winner == model1Player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses
    }

# Model Training

def TrainWithNewGames(name, game, nGames, temperature, batchSize, learningRate, mctsConfig = None):
    game = _resolveGame(game)
    m = Model(name, mctsConfig=mctsConfig)
    examples = _generateTrainingSamples(m, game, nGames, temperature)
    _trainWithExamples(m, examples, batchSize, learningRate)


def _generateTrainingSamples(model, game, nGames, temp, conn=None):
    """ Generates self-play games to learn from.

        This method generates `nGames` self-play games, and returns them as
        a list of `TrainingExample` objects.

        Args:
            `model`: The Blackbird model to use to generate games
            `nGames`: An int determining the number of games to generate.
            `temp`: A float between 0 and 1 determining the exploration temp
                for MCTS. Usually this should be close to 1 to ensure
                high move exploration rate.

        Returns:
            `examples`: A list of `TrainingExample` objects holding all game
                states from the `nGames` games produced.

        Raises:
            ValueError: nGames was not a positive integer.
    """
    if nGames <= 0:
        raise ValueError('Use a positive integer for number of games.')

    examples = []

    for _ in range(nGames):
        gameHistory = []
        state = game()
        lastAction = None
        winner = None
        model.DropRoot()
        while winner is None:
            (nextState, v, currentProbabilties) = model.FindMove(state, temp)
            childValues = model.Root.ChildWinRates()
            example = TrainingExample(state, 1 - v, childValues,
                                      state.EvalToString(
                                          childValues), currentProbabilties, model.Root.Priors,
                                      state.EvalToString(model.Root.Priors), state.LegalActionShape())
            state = nextState
            model.MoveRoot(state)

            winner = state.Winner(lastAction)
            gameHistory.append(example)

        example = TrainingExample(state, None, None, None,
                                  np.zeros(
                                      [len(currentProbabilties)]),
                                  np.zeros(
                                      [len(currentProbabilties)]),
                                  np.zeros(
                                      [len(currentProbabilties)]),
                                  state.LegalActionShape())
        gameHistory.append(example)

        for example in gameHistory:
            if winner == 0:
                example.Reward = 0
            else:
                example.Reward = 1 if example.State.Player == winner else -1
            if conn is not None:
                serialized = state.SerializeState(
                    example.State, example.Probabilities, example.Reward)
                conn.PutGame(state.GameType, serialized)
            else:
                examples.append(example)

    if conn is not None:
        return None
    return examples


def _trainWithExamples(model, examples, batchSize, learningRate, teacher=None):
    """ Trains the neural network on provided example positions.

        Provided a list of example positions, this method will train
        BlackBird's neural network to play better. If `teacher` is provided,
        the neural network will include a cross-entropy term in the loss
        calculation so that the other network's policy is incorporated into
        the learning.

        Args:
            `model`: The Blackbird model to train
            `examples`: A list of `TrainingExample` objects which the
                neural network will learn from.
            `teacher`: An optional `BlackBird` object whose policy the
                current network will include in its loss calculation.
    """
    model.SampleValue.cache_clear()
    model.GetPriors.cache_clear()

    examples = np.random.choice(examples,
                                len(examples) -
                                (len(examples) % batchSize),
                                replace=False)

    for i in range(len(examples) // batchSize):
        start = i * batchSize
        batch = examples[start: start + batchSize]
        model.train(
            np.stack([b.State.AsInputArray()[0] for b in batch], axis=0),
            np.stack([b.Reward for b in batch], axis=0),
            np.stack([b.Probabilities for b in batch], axis=0),
            learningRate,
            teacher
        )
