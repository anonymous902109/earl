from tqdm import tqdm

from citibikes.citibikes_env import CitiBikes
from src.earl.models.bb_models.ppo_model import PPOModel

import numpy as np


def main():
    env = CitiBikes()

    bb_model = PPOModel(env, 'citibikes/trained_models/citibikes')

    n_ep = 100
    from_dict = {i: np.zeros((5, )) for i in range(5)}

    for i in tqdm(range(n_ep)):
        obs, _ = env.reset()
        done = False

        while not done:
            action = bb_model.predict(obs)

            f, to, n = action

            from_dict[f][to] += n

            obs, rew, done, trunc, info = env.step(action)

    for station, vals in from_dict.items():
        vals = [v/sum(vals) for v in vals]
        print('Send from station {} to others: {}'.format(station, vals))








if __name__ == '__main__':
    main()