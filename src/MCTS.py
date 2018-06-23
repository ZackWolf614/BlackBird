import numpy as np
from time import time
import multiprocessing as mp
from GameState import GameState

class Node(object):
    """ This is the abtract tree node class that is used to cache/organize
        game information during the search.
    """
    def __init__(self, state, legalActions, priors, **kwargs):
        self.State = state
        self.Value = 0
        self.Plays = 0
        self.LegalActions = np.array(legalActions)
        self.Children = None
        self.Parents = []
        
        # Use the legal actions mask to ignore priors that don't make sense.
        self.Priors = np.multiply(priors, legalActions)

        # Do some caching here. This is to reduce the strain on the CPU memory
        # cache compared to receating a new array on every access.
        self._childWinRates = np.zeros(len(legalActions))
        self._childPlays = np.zeros(len(legalActions))
        
    def WinRate(self):
        return self.Value/self.Plays if self.Plays > 0 else 0

    def ChildProbability(self):
        allPlays = sum(self.ChildPlays())
        zeroProbs = np.zeros((len(self.ChildPlays())))
        return self.ChildPlays() / allPlays if allPlays > 0 else zeroProbs

    def ChildWinRates(self):
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childWinRates[i] = self.Children[i].WinRate()
        return self._childWinRates

    def ChildPlays(self):
        for i in range(len(self.Children)):
            if self.Children[i] is not None:
                self._childPlays[i] = self.Children[i].Plays
        return self._childPlays

class MCTS(object):
    """ Base class for Monte Carlo Tree Search algorithms. Outlines all the 
        necessary operations for the core algorithm. Most operations will need
        to be overriden to avoid a NotImplemenetedError.
    """
    def __init__(self, explorationRate,
        timeLimit = None, playLimit = None, **kwargs):
        
        self.TimeLimit = timeLimit
        self.PlayLimit = playLimit
        self.ExplorationRate = explorationRate
        self.Root = None
        self.NodeCache = {}
            
    def FindMove(self, state, temp = 0.1, moveTime = None, playLimit = None):
        """ Given a game state, this will use a Monte Carlo Tree Search
            algorithm to pick the best next move. Returns (the chosen state, the
            decided value of input state, and the probabilities of choosing each
            of the children).
        """
        assert isinstance(state, GameState), 'State must inherit from GameState'

        endTime = None
        if moveTime is None:
            moveTime = self.TimeLimit
        if moveTime is not None:
            endTime = time() + moveTime
        if playLimit is None:
            playLimit = self.PlayLimit

        if self.Root is None:
            self.Root = Node(state, state.LegalActions(), self.GetPriors(state))

        assert self.Root.State == state, 'Primed for the correct input state.'
        assert endTime is not None or playLimit is not None, 'MCTS algorithm has a cutoff.'
        
        self._runMCTS(self.Root, temp, endTime, playLimit)

        action = self._selectAction(self.Root, temp, exploring = False)

        return (self._applyAction(state, action), self.Root.WinRate(),
            self.Root.ChildProbability())

    def _runMCTS(self, root, temp, endTime = None, nPlays = None):
        endPlays = root.Plays + (nPlays if nPlays is not None else 0)
        while ((endTime is None or (time() < endTime or root.Children is None))
                and (nPlays is None or root.Plays < endPlays)):
            branch = self.FindLeaf(root, temp)
            node = branch[-1]

            val = self.SampleValue(node.State, node.State.PreviousPlayer)
            self.BackProp(branch, val, node.State.PreviousPlayer)

        return root

    def _selectAction(self, root, temp, exploring = True):
        """ Selects a child of the root using an upper confidence interval. If
            you are not exploring, setting the exploring flag to false will
            instead choose the one with the highest expected payout - ignoring 
            the exploration/regret factor.
        """
        assert root.Children is not None, 'The node has children to select.'
        
        if exploring or temp == 0:
            allPlays = np.sum(root.ChildPlays())
            upperConfidence = (root.ChildWinRates()
                + (self.ExplorationRate * root.Priors * np.sqrt(1.0 + allPlays))
                / (1.0 + root.ChildPlays()))
            choice = np.argmax(upperConfidence)
            p = None
        else:
            if temp < 0.1:
                choice = np.argmax(root.ChildPlays())
                p = None
            else:
                allPlays = np.sum([p**(1/temp) for p in root.ChildPlays()])
                p = [c**(1/temp) / allPlays for c in root.ChildPlays()]
                choice = np.random.choice(len(root.ChildPlays()), p=p)
        
        assert root.LegalActions[choice] == 1, 'Illegal move: \n{}'.format(
            '\n'.join([
                str(root.State),
                str(root.ChildPlays()),
                str(root.Priors),
                str(root.LegalActions),
                str(choice),
                str(p)
            ])
        )
        return choice

    def AddChildren(self, node):
        """ Expands the node and adds children, actions and priors.
        """
        l = len(node.LegalActions)
        node.Children = [None] * l
        for i in range(l):
            if node.LegalActions[i] == 1:
                s = self._applyAction(node.State, i)
                
                node.Children[i] = self.NodeCache.get(s, Node(s, s.LegalActions(), self.GetPriors(s)))
                node.Children[i].Parents.append(node)
        return

    def MoveRoot(self, states):
        """ Function that is used to move the root of the tree to the next
            state. Use this to update the root so that tree integrity can be
            maintained between moves if necessary.
        """
        for s in states: 
            self._moveRoot(s)
        return

    def _moveRoot(self, state):
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
        return

    def DropRoot(self):
        self.Root = None
        self.NodeCache = {}
        return

    def BackProp(self, branch, stateValue, playerForValue):
        for node in branch:
            node.Plays += 1
            if len(node.Parents) > 0:
                if node.Parents[0].State.Player == playerForValue:
                    node.Value += stateValue
                else:
                    node.Value += 1 - stateValue
        return
    
    def _applyAction(self, state, action):
        s = state.Copy()
        s.ApplyAction(action)
        return s

    '''Can override these'''
    '''Algorithm implementation functions'''
    def GetPriors(self, state):
        """ Gets the array of prior search probabilities. 
            Default is just 1 for each possible move.
        """
        return np.array([1] * len(state.LegalActions()))

    def SampleValue(self, state, player):
        """Samples the value of the state for the specified player.
            Must return the value in [0, 1]
            Default is to randomly playout the game.
        """
        rolloutState = state
        winner = rolloutState.Winner()
        while winner is None:
            actions = np.where(rolloutState.LegalActions() == 1)[0]
            action = np.random.choice(actions)
            rolloutState = self._applyAction(rolloutState, action)
            winner = rolloutState.Winner(action)
        return 0.5 if winner == 0 else int(player == winner)

    '''Must override these'''
    def FindLeaf(self, node):
        raise NotImplementedError

    '''Overriden from Object'''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['Pool']
        return self_dict

if __name__=='__main__':
    mcts = MCTS(1, np.sqrt(2))
    print(mcts.TimeLimit)
