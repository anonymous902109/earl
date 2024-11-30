import random

from citibikes.citibikes_env import CitiBikes
from citibikes.fact_generation import get_facts
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_hts import RACCERHTS
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
from src.earl.models.bb_models.ppo_model import PPOModel
from src.earl.utils.util import seed_everything


def main():
    seed_everything(0)
    env = CitiBikes()

    bb_model = PPOModel(env, 'citibikes/trained_models/citibikes', arch=[512, 512], training_timesteps=5e5, lr=0.0005, batch_size=512, verbose=1)

    horizon = 5
    sl_facts, rl_facts = get_facts(env, bb_model, horizon=horizon, perc=0.1, n_states=100)

    domains = list({tuple(bb_model.predict(f.state)) for f in sl_facts}.union({tuple(f.target_action) for f in sl_facts}))

    RACCER_HTS = RACCERHTS(env, bb_model, horizon, n_expand=20, max_level=horizon, n_iter=300)
    RACCER_Advance = NSGARaccerAdvance(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])
    RACCER_Rewind = NSGARaccerRewind(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])

    target_action = random.choice([a for a in domains if a != f.target_action])
    cfs = RACCER_Rewind.explain(rl_facts[0], target=tuple(target_action))