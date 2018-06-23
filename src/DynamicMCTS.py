from MCTS import MCTS, Node
import numpy as np

class DynamicMCTS(MCTS):
    """ An implementation of Monte Carlo Tree Search that aggregates statistics
        as it explores the tree
    """
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)
    
    # Overriding from MCTS
    def FindLeaf(self, node, temp):
        lastAction = None
        branch = [node]
        while True:
            if node.Children is None:
                if node.State.Winner(lastAction) is not None:
                    break
                self.AddChildren(node)
                break
            if np.sum(node.LegalActions) == 0:
                break
            lastAction = self._selectAction(node, temp)
            node = node.Children[lastAction]
            branch.append(node)
            
        return branch


