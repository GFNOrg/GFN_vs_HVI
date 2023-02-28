import numpy as np
import math

from numpy.random import default_rng
from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


class ReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            ('adjacency', np.uint8, (nbytes,)),
            ('num_edges', np.int_, (1,)),
            ('actions', np.int_, (1,)),
            ('is_exploration', np.bool_, (1,)),
            ('delta_scores', np.float_, (1,)),
            ('scores', np.float_, (1,)),
            ('mask', np.uint8, (nbytes,)),
            ('next_adjacency', np.uint8, (nbytes,)),
            ('next_mask', np.uint8, (nbytes,))
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
        self._prev = np.full((capacity,), -1, dtype=np.int_)

    def add(
            self,
            observations,
            actions,
            logs,
            next_observations,
            delta_scores,
            dones,
            indices=None
        ):
        next_indices = np.full((dones.shape[0],), -1, dtype=np.int_)
        if np.all(dones):
            return next_indices

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity
        next_indices[~dones] = add_idx

        data = {
            'adjacency': self.encode(observations['adjacency'][~dones]),
            'num_edges': observations['num_edges'][~dones],
            'actions': actions[~dones],
            'delta_scores': delta_scores[~dones],
            'mask': self.encode(observations['mask'][~dones]),
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones]),

            # Extra keys for monitoring
            'is_exploration': logs['is_exploration'][~dones],
            'scores': observations['score'][~dones],
        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))
        
        if indices is not None:
            self._prev[add_idx] = indices[~dones]

        return next_indices

    def sample(self, batch_size, rng=default_rng()):
        if not self.can_sample(batch_size):
            raise ValueError('Not enough data')  # TODO: Better error message

        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        adjacency = self.decode(samples['adjacency'], dtype=np.int_)
        next_adjacency = self.decode(samples['next_adjacency'], dtype=np.int_)

        # Convert structured array into dictionary
        return {
            'adjacency': adjacency.astype(np.float32),
            'graphs': to_graphs_tuple(adjacency),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'delta_scores': samples['delta_scores'],
            'masks': self.decode(samples['mask']),
            'next_adjacency': next_adjacency.astype(np.float32),
            'next_graphs': to_graphs_tuple(next_adjacency),
            'next_masks': self.decode(samples['next_mask'])
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    @property
    def transitions(self):
        return self._replay[:len(self)]

    def save(self, filename):
        data = {
            'version': 3,
            'replay': self.transitions,
            'index': self._index,
            'is_full': self._is_full,
            'prev': self._prev,
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
            replay._is_full = data['is_full']
            replay._prev = data['prev']
            replay._replay[:len(replay)] = data['replay']
        return replay

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(dtype)

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
        adjacency = np.zeros(shape, dtype=np.float32)
        return {
            'adjacency': adjacency,
            'graphs': graph,
            'num_edges': np.zeros((1,), dtype=np.int_),
            'actions': np.zeros((1,), dtype=np.int_),
            'delta_scores': np.zeros((1,), dtype=np.float_),
            'masks': np.zeros(shape, dtype=np.float32),
            'next_adjacency': adjacency,
            'next_graphs': graph,
            'next_masks': np.zeros(shape, dtype=np.float32)
        }
