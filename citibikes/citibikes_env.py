from datetime import datetime

from gymnasium.spaces import Box
from gymnasium.spaces.multi_discrete import MultiDiscrete
import numpy as np
from maro.simulator.scenarios.citi_bike.common import Action, DecisionType

from maro.simulator import Env

from src.earl.models.envs.abs_env import AbstractEnv


class CitiBikes(AbstractEnv):

    def __init__(self):
        self.gym_env = Env(scenario="citi_bike", topology='toy.5s_6t', start_tick=0, durations=1440, snapshot_resolution=30)

        self.features = ["bikes", "capacity", "fulfillment", "shortage", "failed_return", "trip_requirement",
                         "decision_type", "decision_station_idx", "frame_idx",
                         "holiday", "temperature", "weather", "weekday", "tick"]

        self.station_features = self.features[0:6]
        self.decision_features = self.features[6:9]
        self.shared_features = self.features[9:13]
        self.tick = self.features[13:]

        self.num_stations = len(self.gym_env.current_frame.stations)
        self.max_bike_transfer = 10

        self.state_dim = self.num_stations * len(self.station_features) + \
                         len(self.decision_features) + \
                         len(self.shared_features) + \
                         len(self.tick)

        self.action_space = MultiDiscrete([self.num_stations, self.num_stations, self.max_bike_transfer])
        self.observation_space = Box(low=-np.inf,
                                     high=np.inf,
                                     shape=(self.state_dim,))

        self.max_penalty = sum([station.capacity for station in self.gym_env.current_frame.stations])

        self.decision_types = {
            'NONE': 0,
            'SUPPLY': 1,
            'DEMAND': 2
        }

        self.random_generator = np.random.default_rng(seed=int(datetime.now().timestamp()*10000))

        self.state_feature_names = ['decision_type', 'decision_event_station_idx', 'frame_index'] + \
                                   ['{}_{}'.format(sf, i) for sf in self.station_features for i in range(self.num_stations)] +\
                                   self.shared_features + \
                                   ['tick']

        self.categorical_features = self.state_feature_names
        self.continuous_features = []

        self.state_shape = (self.state_dim,)


    def process_action(self, action):
        if len(action) == 1:
            action = action[0]

        return action

    def step(self, action):
        action = self.process_action(action)
        if action is not None:
            start_station, end_station, number = action
            action = Action(
                from_station_idx=start_station,
                to_station_idx=end_station,
                number=number
            )

        metric, decision_event, self.is_done = self.gym_env.step(action)

        rew = self.calculate_reward(metric, action)

        obs = self.generate_obs(decision_event)

        return np.array(obs), rew, self.is_done, False, {'bike_shortage': metric['bike_shortage']}

    def calculate_reward(self, metric, action):
        num_bikes = 0
        if action is not None:
            num_bikes = action.number

        stations = self.gym_env.current_frame.stations

        bike_shortage = metric['bike_shortage']
        trip_requirements = metric['trip_requirements']
        fulfillment = sum([s.fulfillment for s in stations])

        rew = -(bike_shortage*10.0) / trip_requirements - 0.01 * num_bikes
        return rew

    def reset(self, seed=0):
        self.is_done = False
        self.random_generator = np.random.default_rng(seed=int(datetime.now().timestamp()*100000))
        self.gym_env.reset()
        self.gym_env.step(None)
        obs = self.generate_obs(None)

        return np.array(obs), None

    def generate_obs(self, decision_event):
        obs = []

        stations = self.gym_env.current_frame.stations
        # TODO: probably no sense in including decision_event in state - check without it
        if decision_event is None:
            obs.append(self.decision_types['NONE'])
        elif decision_event.type == DecisionType.Supply:
            obs.append(self.decision_types['SUPPLY'])
        elif decision_event.type == DecisionType.Demand:
            obs.append(self.decision_types['DEMAND'])

        if decision_event is not None:
            obs.append(decision_event.station_idx)
            obs.append(decision_event.frame_index)
        else:
            obs.append(-1)
            obs.append(self.gym_env.frame_index)

        # adding features for each station
        for station_id in range(self.num_stations):
            for f in self.station_features:
                obs.append(getattr(stations[station_id], f))

        # adding features same for all stations such as weather
        for f in self.shared_features:  # tick cannot be added this way
            obs.append(getattr(stations[0], f))

        obs.append(self.gym_env.tick)

        return obs

    def render(self):
        print(self.gym_env.summary)

    def close(self):
        pass

    def get_actions(self, x=None):
        if x is None:
            actions = [(i, j, k)
                       for i in range(self.action_space[0].n)
                       for j in range(self.action_space[1].n)
                       for k in range(self.action_space[2].n)]

            return actions

    def set_nonstoch_state(self, state, env_state=None):
        for s_id in range(self.num_stations):
            station_info = state[s_id]
            for var_name, var_val in station_info.items():
                setattr(self.gym_env.current_frame.stations[s_id], var_name, var_val)

        self.state = self.generate_obs(None)

    def get_state(self):
        state = {}
        for s_id in range(self.num_stations):
            station_info = {}
            station = self.gym_env.current_frame.stations[s_id]
            variables = [a for a in dir(station) if not a.startswith('__')]
            for v in variables:
                station_info[v] = getattr(station, v)

            state[s_id] = station_info

        return state

    def get_env_state(self):
        return None
