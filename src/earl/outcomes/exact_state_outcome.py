

class ExactStateOutcome():

    def __init__(self, state):
        self.state = list(state)
        self.name = 'exact state = {}'.format(state)

    def equal_states(self, x):
        return (self.state == list(x))
