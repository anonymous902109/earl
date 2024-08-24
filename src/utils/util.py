import copy
import os
import pickle
import random
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from src.models.trajectory import Trajectory


def seed_everything(seed):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)


def generate_facts_with_action_outcome(env, bb_model, path, outcome, horizon=5, n_episodes=10000, max_traj=100):
    '''
     Generates a datasets of Trajectory objects with a specified outcome.
    '''
    try:
        path = os.path.join(path, outcome.name)
        with open(path + '.pkl', 'rb') as f:
            trajectories = pickle.load(f)

        for t in trajectories:
            t.set_outcome(outcome)

        return trajectories

    except FileNotFoundError:
        trajectories = []
        print('Generating facts for outcome {}'.format(outcome.name))
        traj_id = 0

        for ep_id in tqdm(range(n_episodes)):
            obs, _ = env.reset(int(datetime.now().timestamp() * 100000))
            done = False
            outcome_found = False
            t = Trajectory(traj_id, horizon)

            if traj_id >= max_traj:
                break

            while not done:
                action = bb_model.predict(obs)
                t.append(copy.copy(obs), action, copy.deepcopy(env.get_env_state()))

                if (outcome.explain_outcome(env, obs)) and not outcome_found:  # if outcome should be explained
                    if ((t.num_actions() - 1) >= horizon):  # if there are enough previous states
                        t.mark_outcome_state()
                        outcome_found = True

                if outcome_found:
                    # check that the same fact hasn't been added before
                    if t.states[t.outcome_id].tolist() not in [prev_t.states[t.outcome_id].tolist() for prev_t in trajectories]:
                        trajectories.append(t)
                        traj_id += 1

                    t = Trajectory(traj_id, horizon)
                    outcome_found = False

                new_obs, rew, done, trunc, info = env.step(action)
                done = done or trunc

                obs = new_obs

        # save without outcome as it cannot be pickled
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

    for t in trajectories:
        t.set_outcome(outcome)

    return trajectories

