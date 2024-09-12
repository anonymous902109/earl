from src.earl.algorithms.carla.carla_alg_factory import CarlaAlgFactory


class CarlaCFS:

    def __init__(self, env, bb_model, alg='', hyperparams={}):
        self.alg = CarlaAlgFactory.get_algorithm(bb_model, hyperparams, alg)
        self.hyperparams = hyperparams

    def get_best_cf(self, fact, target):
        cfs = self.alg.get_counterfactuals(fact)

        return cfs
