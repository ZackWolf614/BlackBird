import numpy as np
from time import time
import multiprocessing as mp
from GameState import GameState


class Node(object):
    """ Base class for storing game state information in tree searches.

        This is the abtract tree node class that is used to cache/organize game
        information during the search.

        Attributes:
            State: A GameState object holding the Node's state representation.
            Value: A float holding the Node's state valuation.
            Plays: A counter holding the number of times the Node has been used.
            LegalActions: An int holding the number of legal actions for the
                Node.
            Children: A list of Nodes holding all legal states for the Node.
            Parent: A Node object representing the Node's parent.
            Priors: A numpy array of size [num_legal_actions] that holds the
                Node's prior probabilities.  At instantiation, the provided
                prior is filtered on only legal moves.

            _childWinRates: A numpy array of size [num_legal_actions] used for
                storing the win rates of the Node's children in MCTS.
            _childPlays: A numpy array of size [num_legal_actions] used for
                storing the play counts of the Node's children in MCTS.
    """

    def __init__(self, state, legalActions, priors, **kwargs):
        self.State = state
        self.Value = 0
        self.Plays = 0
        self.LegalActions = np.array(legalActions)
        self.Children = None
        self.Parent = None
        self.Priors = np.multiply(priors, legalActions)

        self._childWinRates = np.zeros(len(legalActions))
        self._childPlays = np.zeros(len(legalActions), dtype=np.float)

    def WinRate(self):
        """ Samples the win rate of the Node after MCTS.

            This is a simple API which provides the win rate of the Node after
            applying MCTS.

            Returns:
                A float representing the win rate for the Node. If no plays have
                been applied to this Node, the default value of 0 is returned.
        """
        return self.Value/self.Plays if self.Plays > 0 else 0

    def ChildProbability(self):
        """ Samples the probabilities of sampling each child Node.

            Samples the play rate for each of the Node's children Node objects.
            If no children have been sampled in MCTS, this returns zeros.

            Returns:
                A numpy array representing the play rate for each of the Node's
                children. Defaults to an array of zeros if no children have been
                sampled.
        """
        allPlays = sum(self.ChildPlays())
        zeroProbs = np.zeros((len(self.ChildPlays())), dtype=np.float)
        return self.ChildPlays() / allPlays if allPlays > 0 else zeroProbs

    def ChildWinRates(self):
        """ Samples the win rate of each child Node object.

            Samples the win rates for each of the Node's children. Not helpful
            if none of the children have been evaluated in MCTS.

            Returns:
                A numpy array representing the win rate for each of the Node's
                children.
        """
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childWinRates[i] = self.Children[i].WinRate()
        return self._childWinRates

    def ChildPlays(self):
        """ Samples the play rate of each child Node object.

            Samples the play rates for each of the Node's children. Not helpful
            if none of the children have been evaluated in MCTS.

            Returns:
                A numpy array representing the play rate for each of the Node's
                children.
        """
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childPlays[i] = self.Children[i].Plays
        return self._childPlays


class MCTS(object):
    """ Base class for Monte Carlo Tree Search algorithms.

        Outlines all the necessary operations for the core MCTS algorithm.
        _findLeaf() will need to be overriden to avoid a NotImplemenetedError.

        Attributes:
            TimeLimit: The default max move time in seconds.
            PlayLimit: The default number of positions to evaluate per move.
            ExplorationRate: The exploration parameter for MCTS.
            Root: The Node object representing the root of the MCTS.
    """

    def __init__(self, explorationRate, timeLimit=None, playLimit=None,
                 **kwargs):

        self.TimeLimit = timeLimit
        self.PlayLimit = playLimit
        self.ExplorationRate = explorationRate
        self.Root = None

    def AddChildren(self, node):
        """ Expands a node and adds children, actions and priors.

            Given a node, MCTS will evaluate the node's children, if they exist.
            The evaluation and prior policy are supplied in the creation of the
            child Node object.

            Args:
                node: A Node object to expand.
        """
        numLegalMoves = len(node.LegalActions)
        node.Children = [None] * numLegalMoves
        for actionIndex in range(numLegalMoves):
            if node.LegalActions[actionIndex] == 1:
                s = self._applyAction(node.State, actionIndex)
                node.Children[actionIndex] = Node(s, s.LegalActions(),
                                                  self.GetPriors(s))
                node.Children[actionIndex].Parent = node

    def DropRoot(self):
        """ Resets self.Root to None
        """
        self.Root = None

    def FindMove(self, state, temp=0.1, moveTime=None, playLimit=None):
        """ Finds the optimal move in a position.

            Given a game state, this will use a Monte Carlo Tree Search
            algorithm to pick the best next move.

            Args:
                state: A GameState object which the function will evaluate.
                temp: A float determining the temperature to apply in move
                    selection.
                moveTime: An optional float determining the allowed search time.
                playLimit: An optional float determining the allowed number of
                    positions to evaluate.

            Returns:
                A tuple providing, in order...
                    - The board state after applying the selected move
                    - The decided value of input state
                    - The probabilities of choosing each of the children

            Raises:
                TypeError: state was not an object of type GameState.
                ValueError: The function was not able to determine a stop time.
        """
        if not isinstance(state, GameState):
            raise TypeError('State not of type GameState')

        endTime = None
        if moveTime is None:
            moveTime = self.TimeLimit
        if moveTime is not None:
            endTime = time() + moveTime
        if playLimit is None:
            playLimit = self.PlayLimit

        if endTime is None and playLimit is None:
            raise ValueError('Not enough information to decide a stop time.')

        if self.Root is None:
            self.Root = Node(state, state.LegalActions(),
                             self.GetPriors(state))
        assert self.Root.State == state, 'Primed for the correct input state.'

        self._runMCTS(temp, endTime, playLimit)
        action = self._selectAction(self.Root, temp, exploring=False)

        return (self._applyAction(state, action), self.Root.WinRate(),
                self.Root.ChildProbability())

    def MoveRoot(self, state):
        """ This is the public API of MCTS._moveRoot.

            Move the root of the tree to the provided state. Use this to update
            the root so that tree integrity can be maintained between moves if
            necessary. Does nothing if Root is None, for example after running
            DropRoot().

            Args:
                state: A GameState object which self.Root should be updated to.
        """
        self._moveRoot(state)

    def ResetRoot(self):
        """ Set self.Root to the appropriate initial state.

            Reset the state of self.Root to an appropriate initial state.  If
            self.Root was already None, then there is nothing to do, and it will
            remain None.  Otherwise, ResetRoot will apply an iterative backup to
            self.Root until its parent is None.
        """
        if self.Root is None:
            return
        while self.Root.Parent is not None:
            self.Root = self.Root.Parent

    def _applyAction(self, state, action):
        """ Applies an action to a provided state.

            Args:
                state: A GameState object which needs to be updated.
                action: An int which indicates the action to apply to state.
        """
        s = state.Copy()
        s.ApplyAction(action)
        return s

    def _backProp(self, leaf, stateValue, playerForValue):
        """ Backs up a value from a leaf through to self.Root.

            Given a leaf node and a value, this function will back-propogate the
            value to its parent node, and propogate that all the way through the
            tree to its root, self.Root

            Args:
                leaf: A Node object which is the leaf of the current tree to
                    apply back-propogation to.
                stateValue: The MCTS-created evaluation to back-propogate.
                playerForValue: The player which stateValue applies to.
        """
        leaf.Plays += 1
        if leaf.Parent is not None:
            if leaf.Parent.State.Player == playerForValue:
                leaf.Value += stateValue
            else:
                leaf.Value += 1 - stateValue

            self._backProp(leaf.Parent, stateValue, playerForValue)

    def _moveRoot(self, state):
        """ Updates the root of the tree.

            Move the root of the tree to the provided state. Use this to update
            the root so that tree integrity can be maintained between moves if
            necessary. Does nothing if Root is None, for example after running
            DropRoot().

            Args:
                state: A GameState object which self.Root should be updated to.
        """
        if self.Root is None:
            return
        if self.Root.Children is None:
            self.Root = None
            return
        for child in self.Root.Children:
            if child is None:
                continue
            if child.State == state:
                self.Root = child
                break

    def _runMCTS(self, temp, endTime=None, nPlays=None):
        """ Run the MCTS algorithm on the current Root Node.

            Given the current game state, represented by self.Root, a child node
            is seleted using the _findLeaf method. This method will apply temp to
            all child node move selection proportions, compute the sampled value
            of the action, and backpropogate the value through the tree.

            Args:
                temp: A float determining the temperature to apply in FindMove.
                endTime: (optional) The maximum time to spend on searching.
                nPlays: (optional) The maximum number of positions to evaluate.
        """
        endPlays = self.Root.Plays + (nPlays if nPlays is not None else 0)
        while ((endTime is None or (time() < endTime or self.Root.Children is None))
                and (nPlays is None or self.Root.Plays < endPlays)):
            node = self._findLeaf(self.Root, temp)

            val = self.SampleValue(node.State, node.State.PreviousPlayer)
            self._backProp(node, val, node.State.PreviousPlayer)

    def _selectAction(self, root, temp, exploring=True):
        """ Chooses an action from an explored root.

            Selects a child of the root using an upper confidence interval. If
            you are not exploring, setting the exploring flag to false will
            instead choose the one with the highest expected payout - ignoring 
            the exploration/regret factor.

            Args:
                root: A Node object which must have children Nodes.
                temp: The temperature to apply to the children Node visit
                    counts. If temp is 0, _selectAction will return the child
                    Node with the greatest visit count.
                exploring: A boolean toggle for overriding the selection type to
                    a simple argmax.  If True, _selectAction will return the
                    child Node with the greatest visit count.

            Returns:
                choice: An int representing the index of the selected action.
        """
        assert root.Children is not None, 'The node has children to select.'

        if exploring or temp == 0:
            allPlays = sum(root.ChildPlays())
            upperConfidence = (root.ChildWinRates()
                               + (self.ExplorationRate *
                                  root.Priors * np.sqrt(1.0 + allPlays))
                               / (1.0 + root.ChildPlays()))
            choice = np.argmax(upperConfidence)
            p = None
        else:
            allPlays = sum([p ** (1 / temp) for p in root.ChildPlays()])
            p = [c ** (1 / temp) / allPlays for c in root.ChildPlays()]
            choice = np.random.choice(len(root.ChildPlays()), p=p)

        assert root.LegalActions[choice] == 1, 'Selected move is legal.'
        return choice

    '''Functions to override'''

    def GetPriors(self, state):
        """ Gets the array of prior search probabilities.

            This is the default GetPriors for MCTS. The return value is always
            an array of ones. This should be overridden to get actual utility.

            Args:
                state: A GameState object to get the priors of.

            Returns:
                A numpy array of ones of shape [num_legal_actions_of_state].
        """
        return np.array([1] * len(state.LegalActions()))

    def SampleValue(self, state, player):
        """ Samples the value of a state for a specified player.

            This applies a set of Monte Carlo random rollouts to a state until a
            game terminates, and returns the determined evaluation.

            Args:
                state: A GameState object which the function will obtain the
                    evaluation of.
                player: An integer representing the current player in state.

            Returns:
                A float representing the value of the state. It is 0 if it was
                    determined to be a loss, 1 if it was determined to be a win,
                    and 0.5 if it was determined to be a draw.
        """
        rolloutState = state
        winner = rolloutState.Winner()
        while winner is None:
            actions = np.where(rolloutState.LegalActions() == 1)[0]
            action = np.random.choice(actions)
            rolloutState = self._applyAction(rolloutState, action)
            winner = rolloutState.Winner(action)
        return 0.5 if winner == 0 else int(player == winner)

    def _findLeaf(self, node, temp):
        """ Applies MCTS to a supplied node until a leaf is found.

            Args:
                node: A Node object to find a leaf of.
        """
        raise NotImplementedError

    '''Overriden from Object'''

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['Pool']
        return self_dict


if __name__ == '__main__':
    mcts = MCTS(1, np.sqrt(2))
    print(mcts.TimeLimit)
