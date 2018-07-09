from NetworkFactory import NetworkFactory
from DynamicMCTS import DynamicMCTS as MCTS
from Network import Network
import uuid
import json
import os
import functools
import numpy as np


class Model(MCTS, Network):
    ModelDirectory = 'blackbird_models'
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

    def __init__(self, name, version=None, mctsConfig=None, networkConfig=None, tensorflowConfig=None):
        self.Name = name
        self.Version = version if version is not None else Model.LatestVersion(name)

        self.UUID = self._loadUUID()

        self.MCTSConfig = mctsConfig if mctsConfig is not None else self._loadMCTS()
        print(json.dumps(self.MCTSConfig))
        self.TensorflowConfig = tensorflowConfig if tensorflowConfig is not None else self._loadTF()

        if networkConfig is None:
            networkConfig = self.versionDir

        MCTS.__init__(self, **self.MCTSConfig)
        Network.__init__(self, self.versionDir, NetworkFactory(networkConfig), self.TensorflowConfig)

    @classmethod
    def LatestVersion(cls, name):
        latest = 0
        for folder in os.listdir(os.path.join(Model.ModelDirectory, name)):
            if os.path.isdir(folder):
                version = int(folder)
                if version > latest:
                    latest = version
        return latest

    def _loadUUID(self):
        uuidf = os.path.join(self.parentDir, '.uuid')
        if os.path.exists(uuidf):
            with open(uuidf, 'r') as fin:
                return fin.read()
        else:
            id = uuid.uuid4().hex
            with open(uuidf, 'w') as fout:
                fout.write(id)
            return id

    def _loadMCTS(self):
        mctsf = os.path.join(self.parentDir, 'mctsConfig.json')
        with open(mctsf, 'r') as fin:
            return json.load(fin)
        
    def SaveMCTS(self):
        mctsf = os.path.join(self.parentDir, 'mctsConfig.json')
        with open(mctsf, 'w') as fin:
            return json.dump(self.MCTSConfig, fin)

    def _loadTF(self):
        tff = os.path.join(self.parentDir, 'tensorflowConfig.json')
        with open(tff, 'r') as fin:
            return json.load(fin)
            
    def SaveTF(self):
        tff = os.path.join(self.parentDir, 'tensorflowConfig.json')
        with open(tff, 'w') as fin:
            return json.dump(self.TensorflowConfig, fin)

    def SaveModel(self):
        self.saveModel(self.versionDir)

    def SaveMCTS(self):
        mctsf = os.path.join(self.parentDir, 'mctsConfig.json')
        with open(mctsf, 'w') as fin:
            return json.dump(self.MCTSConfig, fin)

    def LastVersion(self):
        return Model(self.Name, self.MCTSConfig)

    @property
    def parentDir(self):
        return os.path.join(Model.ModelDirectory, self.Name)

    @property
    def versionDir(self):
        return os.path.join(self.parentDir, str(self.Version))

    @property
    def Name(self):
        return self._name

    @Name.setter
    def Name(self, value):
        self._name = value
        if not os.path.exists(self.parentDir):
            os.makedirs(self.parentDir)

    @property
    def Version(self):
        return self._version

    @Version.setter
    def Version(self, value):
        self._version = value
        if not os.path.exists(self.versionDir):
            os.makedirs(self.versionDir)

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
