from src.outcomes.abstract_outcome import AbstractOutcome


class OneActionOutcome(AbstractOutcome):

    def __init__(self, bb_model=None, target_action=None):
        super(OneActionOutcome, self).__init__(bb_model, target_action)

        self.outcome = target_action

        self.name = 'why not {}'.format(self.target_action)

    def cf_outcome(self, state):
        a = self.target_action == self.bb_model.predict(state)
        return a  # counterfactual where one specific action is required.

    def sf_outcome(self, state):
        a = self.target_action != self.bb_model.predict(state)
        return a  # semifactual where one specific action is required -- even if ... still not target action

    def explain_outcome(self, env, state=None):

        if (not env.is_done) and (self.bb_model.predict(state) == self.target_action):
            return True

        return False