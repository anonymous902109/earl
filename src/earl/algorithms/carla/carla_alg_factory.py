class CarlaAlgFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_algorithm(bb_model, hyperparams, alg=''):
        if alg == 'growing_spheres':
            return GrowingSpheres(bb_model, hyperparams)
        else:
            raise TypeError('Algorithm with name {} is not supported. Supported CARLA algorithms are: .'.format(alg))