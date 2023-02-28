import numpy as np


def multirange(counts):
    # https://stackoverflow.com/a/20033438/6826667
    reset_idx = np.cumsum(counts[:-1])
    results = np.ones(np.sum(counts), dtype=np.int_)
    results[0] = 0
    results[reset_idx] = 1 - counts[:-1]
    results.cumsum(out=results)
    return results


def dense_multirange_with_offset(lengths, max_length):
    batch_size = lengths.shape[0]
    masks = (lengths < max_length)

    indices = np.ones((batch_size, max_length), dtype=np.int_)
    indices[:, 0] = 0
    indices[masks, lengths[masks]] = -max_length - 1
    indices = np.cumsum(indices, axis=1)

    offsets = np.zeros((batch_size, 1), dtype=np.int_)
    offsets[1:] = lengths[:-1, None]
    offsets = np.cumsum(offsets, axis=0)

    return np.where(indices >= 0, indices + offsets, -1)
