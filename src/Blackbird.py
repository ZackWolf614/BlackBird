from DynamicMCTS import DynamicMCTS as MCTS
from RandomMCTS import RandomMCTS
from FixedMCTS import FixedMCTS
from Network import Network
from NetworkFactory import NetworkFactory

import functools
import random
import yaml
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)


class TrainingExample(object):
    def __init__(self, state, value, childValues, childValuesStr,
                 probabilities, priors, priorsStr, boardShape):

        self.State = state
        self.Value = value
        self.BoardShape = boardShape
        self.ChildValues = childValues if childValues is not None else None
        self.ChildValuesStr = childValuesStr if childValuesStr is not None else None
        self.Reward = None
        self.Priors = priors
        self.PriorsStr = priorsStr
        self.Probabilities = probabilities

    def __str__(self):
        state = str(self.State)
        value = 'Value: {}'.format(self.Value)
        childValues = 'Child Values: \n{}'.format(self.ChildValuesStr)
        reward = 'Reward:\n{}'.format(self.Reward)
        probs = 'Probabilities:\n{}'.format(
            self.Probabilities.reshape(self.BoardShape))
        priors = '\nPriors:\n{}\n'.format(self.PriorsStr)

        return '\n'.join([state, value, childValues, reward, probs, priors])


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


def GenerateTrainingSamples(model, nGames, temp, conn=None):
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
        state = model.Game()
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
                serialized = state.SerializeState(example.State, example.Probabilities, example.Reward)
                conn.PutGame(state.GameType, serialized)

        examples += gameHistory

    return examples


def TrainWithExamples(model, examples, batchSize, learningRate, teacher=None):
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


class Model(MCTS, Network):
    """ Class which encapsulates MCTS powered by a neural network.

        The BlackBird class is designed to learn how to win at a board game, by
        using Monte Carlo Tree Search (MCTS) with the tree search powered by a
        neural network.
        Args:
            `game`: A GameState object which holds the rules of the game
                BlackBird is intended to learn.
            `name`: The name of the model.
            `mctsConfig` : JSON config for MCTS runtime evaluation
            `networkConfig` : JSON config for creating a new network from NetworkFactory
            `tensorflowConfig` : Configuaration for tensorflow initialization
    """

    def __init__(self, game, name, mctsConfig, networkConfig={}, tensorflowConfig={}):
        self.Game = game
        self.Name = name
        self.MCTSConfig = mctsConfig
        self.NetworkConfig = networkConfig
        self.TensorflowConfig = tensorflowConfig
        MCTS.__init__(self, **mctsConfig)

        if networkConfig != {}:
            Network.__init__(self, name, NetworkFactory(
                networkConfig), tensorflowConfig)
        else:
            Network.__init__(self, name, tensorflowConfig=tensorflowConfig)

    def LastVersion(self):
        return Model(self.Game, self.Name, self.MCTSConfig, self.NetworkConfig, self.TensorflowConfig)

    @functools.lru_cache(maxsize=4096)
    def SampleValue(self, state, player):
        """ Returns BlackBird's evaluation of a supplied position.

            BlackBird's network will evaluate a supplied position, from the
            perspective of `player`.

            Args:
                `state`: A GameState object which should be evaluated.
                `player`: An int representing the current player.

            Returns:
                `value`: A float between 0 and 1 holding the evaluation of the
                    position. 0 is the worst possible evaluation, 1 is the best.
        """
        value = self.getEvaluation(state.AsInputArray())
        value = (value + 1) * 0.5  # [-1, 1] -> [0, 1]
        if state.Player != player:
            value = 1 - value
        assert value >= 0, 'Value: {}'.format(value)
        return value

    @functools.lru_cache(maxsize=4096)
    def GetPriors(self, state):
        """ Returns BlackBird's policy of a supplied position.

            BlackBird's network will evaluate the policy of a supplied position.

            Args:
                `state`: A GameState object which should be evaluated.

            Returns:
                `policy`: A list of floats of size `len(state.LegalActions())` 
                    which sums to 1, representing the probabilities of selecting 
                    each legal action.
        """
        policy = self.getPolicy(state.AsInputArray()) * state.LegalActions()
        policy /= np.sum(policy)

        return policy
