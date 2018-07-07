class GameState(object):
    def __init__(self):
        self.Board = None
        self.Player = None
        self.PreviousPlayer = None

    def Copy(self):
        raise NotImplementedError

    def LegalActions(self):
        raise NotImplementedError

    def LegalActionShape(self):
        raise NotImplementedError

    def ApplyAction(self, action):
        raise NotImplementedError

    def Winner(self, prevAction=None):
        raise NotImplementedError

    def NumericRepresentation(self):
        raise NotImplementedError

    def EvalToString(self, eval):
        return str(eval)

    def SerializeState(self, state, policy, eval):
        raise NotImplementedError

    def DeserializeState(self, serialState):
        raise NotImplementedError
