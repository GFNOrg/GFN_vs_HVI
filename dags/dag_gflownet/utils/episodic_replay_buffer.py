import numpy as np
import math

from jraph import GraphsTuple
from numpy.random import default_rng

from dag_gflownet.utils.jraph_utils import batch_sequences_to_graphs_tuple


class EpisodicReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        self.max_length = (num_variables * (num_variables - 1) // 2) + 1
        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            ('actions', np.int_, (self.max_length,)),
            ('delta_scores', np.float_, (self.max_length,)),
            ('is_exploration', np.bool_, (self.max_length, 1)),
            ('mask', np.uint8, (self.max_length, nbytes)),
            ('log_pi', np.float32, (self.max_length,)),
            ('is_complete', np.bool_, ()),
            ('length', np.int_, ()),
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0

    def add(self, observations, actions, logs, next_observations, delta_scores, dones, indices=None):
        if indices is None:
            indices = self._index + np.arange(actions.shape[0])
            self._index += actions.shape[0]

        # Add data to the episode buffer
        episode_idx = self._replay['length'][indices]
        self._replay['actions'][indices, episode_idx] = actions
        self._replay['delta_scores'][indices, episode_idx] = delta_scores
        self._replay['is_exploration'][indices, episode_idx] = logs['is_exploration']
        self._replay['mask'][indices, episode_idx] = self.encode(observations['mask'])
        self._replay['log_pi'][indices, episode_idx] = logs['log_pi']

        # Set complete episodes & update episode indices
        self._replay['is_complete'][indices[dones]] = True
        self._replay['length'][indices[~dones]] += 1

        # Get new indices for new trajecories, and clear data already present
        num_dones = np.sum(dones)
        new_indices = (self._index + np.arange(num_dones)) % self.capacity
        self._replay[new_indices] = 0  # Clear data

        # Set the new indices for the next trajectories
        indices[dones] = new_indices
        self._index = (self._index + num_dones) % self.capacity

        return indices

    def sample(self, batch_size, rng=default_rng()):
        if not self.can_sample(batch_size):
            raise ValueError('Not enough data')  # TODO: Better error message

        indices = rng.choice(self.capacity, batch_size,
            replace=False, p=self._replay['is_complete'] / len(self))
        samples = self._replay[indices]

        lengths, actions = samples['length'], samples['actions']
        graphs = batch_sequences_to_graphs_tuple(
            self.num_variables, actions, lengths)

        return {
            'graphs': graphs,
            'masks': self.decode(samples['mask']),
            'actions': actions,
            'scores': np.sum(samples['delta_scores'], axis=1),
            'num_edges': lengths,
            'log_pi': samples['log_pi'],
        }

    def __len__(self):
        return np.sum(self._replay['is_complete'])

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def save(self, filename):
        data = {
            'version': 3,
            'replay': self._replay,
            'index': self._index,
            'capacity': self.capacity,
            'num_variables': self.num_variables,
        }
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            if data['version'] != 3:
                raise IOError(f'Unknown version: {data["version"]}')
            replay = cls(
                capacity=data['capacity'],
                num_variables=data['num_variables']
            )
            replay._index = data['index']
            replay._replay = data['replay']
        return replay

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        shape = (*encoded.shape[:-1], self.num_variables, self.num_variables)
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        return decoded.reshape(*shape).astype(dtype)

    @property
    def dummy(self):
        shape = (1, self.num_variables, self.num_variables)
        graph = GraphsTuple(
            nodes=np.arange(self.num_variables),
            edges=np.zeros((1,), dtype=np.int_),
            senders=np.zeros((1,), dtype=np.int_),
            receivers=np.zeros((1,), dtype=np.int_),
            globals=None,
            n_node=np.full((1,), self.num_variables, dtype=np.int_),
            n_edge=np.ones((1,), dtype=np.int_),
        )
        return {
            'graphs': graph,
            'masks': np.zeros(shape, dtype=np.float32),
            'actions': np.zeros((1,), dtype=np.int_),
            'scores': np.zeros((1,), dtype=np.float_),
            'num_edges': np.ones((1,), dtype=np.int_),
            'log_pi': np.zeros((1,), dtype=np.float32),
        }


class OnPolicyEpisodicReplayBuffer(EpisodicReplayBuffer):
    def reset(self):
        self._replay[:] = 0  # Clear data
        return 0

    def add(self, observations, actions, logs, delta_scores, dones, indices):
        assert actions.shape[0] == self.capacity

        self._replay['actions'][:, indices] = actions
        self._replay['delta_scores'][:, indices] = delta_scores
        self._replay['is_exploration'][:, indices] = logs['is_exploration']
        self._replay['mask'][:, indices] = self.encode(observations['mask'])
        self._replay['log_pi'][:, indices] = logs['log_pi']

        # Set complete episodes & update episode indices
        self._replay['is_complete'][dones] = True
        self._replay['length'][~dones] += 1

        return indices + 1

    def sample(self, batch_size, **kwargs):
        if not self.can_sample(batch_size):
            raise ValueError('Not enough data')  # TODO: Better error message

        lengths, actions = self._replay['length'], self._replay['actions']
        graphs = batch_sequences_to_graphs_tuple(
            self.num_variables, actions, lengths)

        return {
            'graphs': graphs,
            'masks': self.decode(self._replay['mask']),
            'actions': actions,
            'scores': np.sum(self._replay['delta_scores'], axis=1),
            'num_edges': lengths,
            'log_pi': self._replay['log_pi'],
        }

    def can_sample(self, batch_size):
        assert batch_size == self.capacity
        return np.all(self._replay['is_complete'])
