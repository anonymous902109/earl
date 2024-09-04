from src.earl.outcomes.abstract_outcome import AbstractOutcome


class ActionOutcome(AbstractOutcome):

    def __init__(self, bb_model, target_action=None, true_action=None):
        super(ActionOutcome, self).__init__(bb_model, target_action, true_action)

        self.name = 'change_{}_to_any'.format(true_action)# TODO: insert human-readable here

    def cf_outcome(self, env, state):
        return self.true_action != self.bb_model.predict(state)  # counterfactual where any alternative action is chosen

    def explain_outcome(self, env, state=None):
        if self.bb_model.predict(state) == self.true_action:
            return True

        return False