import numpy as np
import pandas as pd
from tqdm import tqdm


class CustomDataset:

    def __init__(self, env, bb_model, dataset_path, k=10):
        self.env = env
        self.bb_model = bb_model
        self.dataset_path = dataset_path

        self._dataset = self.generate_dataset(env, bb_model, dataset_path, n_ep=1000, k=k)

    def generate_dataset(self, env, model, dataset_path, n_ep=100, k=10):
        try:
            df = pd.read_csv(dataset_path, index_col=False)
            print('Loaded datasets with {} samples'.format(len(df)))
        except FileNotFoundError:
            print('Generating datasets...')
            ds_len_k = []

            for i in tqdm(range(n_ep)):
                obs, _ = env.reset()
                done = False
                c = 0
                ds = []
                actions = []
                while not done:
                    c += 1
                    ds.append(list(obs))
                    rand = np.random.randint(0, 2)
                    if rand == 0:
                        action = model.predict(obs)
                    else:
                        action = np.random.choice(env.get_actions(obs))

                    actions.append(action)
                    obs, rew, done, trunc,  info = env.step(action)

                ds.append(list(obs))

                # generate a set of trajectories of len k
                if c >= k:
                    for l in range(c - k - 1):
                        ds_len_k.append(list(np.array(ds[l:l+k+1]).flatten()) + actions[l:l+k])  # append observations and actions

            df = pd.DataFrame(ds_len_k)
            df = df.drop_duplicates()

            print('Generated {} samples!'.format(len(df)))
            df.to_csv(dataset_path, index=False)

        return df

    def split_dataset(self, frac=0.8):
        train_dataset = self._dataset.sample(frac=0.8, random_state=1)
        test_dataset = self._dataset.drop(train_dataset.index)

        return train_dataset, test_dataset