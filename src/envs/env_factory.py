from src.envs.farmenv import FarmEnv
from src.envs.frozen_lake import FrozenLake
from src.envs.gridworld import Gridworld
from src.envs.highway_env import HighwayEnv


class EnvFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_env(task_name, params):
        if task_name == "gridworld":
            env = Gridworld(params)
        elif task_name == "frozen_lake":
            env = FrozenLake(params)
        elif task_name == "highway":
            env = HighwayEnv()
        elif task_name == "farm0":
            env = FarmEnv()
        # elif task_name == "citibikes":
        #     # TODO: implement citibikes
        #     env = CitiBikes()
        else:
            raise NotImplementedError('Env with name {} is not supported'.format(task_name))

        return env