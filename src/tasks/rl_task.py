import copy
from datetime import datetime

from tqdm import tqdm

from src.models.trajectory import Trajectory
from src.tasks.task import Task


class RLTask(Task):

    def __init__(self):
        pass

    def generate_facts_with_outcome(self, env, bb_model, outcome, horizon, max_traj, n_episodes):
        print('Generating facts for outcome {}'.format(outcome.name))
        traj_id = 0
        trajectories = []

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
                    if t.states[t.outcome_id].tolist() not in [prev_t.states[t.outcome_id].tolist() for prev_t in
                                                               trajectories]:
                        trajectories.append(t)
                        traj_id += 1

                    t = Trajectory(traj_id, horizon)
                    outcome_found = False

                new_obs, rew, done, trunc, info = env.step(action)
                done = done or trunc

                obs = new_obs

        for t in trajectories:
            t.set_outcome(outcome)

        return trajectories