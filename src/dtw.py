import numpy as np
from scipy.spatial.distance import cdist
from numba import jit

@jit(nopython=True, cache=True)
def accumulate_cost(C, D, steps, step_sizes, weights_mul, weights_add, max_0, max_1):
    for n in range(max_0, D.shape[0]):
        for m in range(max_1, D.shape[1]):
            for idx, w_add, w_mul in zip(range(step_sizes.shape[0]), weights_add, weights_mul):
                prev_D = D[n - step_sizes[idx, 0], m - step_sizes[idx, 1]]
                curr_C = w_mul * C[n - max_0, m - max_1] + w_add
                cost = prev_D + curr_C
                if cost < D[n, m]:
                    D[n, m] = cost
                    steps[n, m] = idx
    return D, steps

@jit(nopython=True, cache=True)
def backtrack_path(steps, step_sizes, start):
    pos = (steps.shape[0] - 1, start)
    path = [(pos[0], pos[1])]
    while pos[0] > 0:
        step_idx = steps[pos[0], pos[1]]
        pos = (pos[0] - step_sizes[step_idx][0], pos[1] - step_sizes[step_idx][1])
        if min(pos) < 0:
            break
        path.append((pos[0], pos[1]))
    return path

def dtw(X, Y):
    step_sizes = np.array([[1,1], [0,1], [1,0]], dtype=np.uint32)
    weights_add = np.zeros(3, dtype=np.float64)
    weights_mul = np.ones(3, dtype=np.float64)

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # Validate feature dimensions
    if X.shape[0] != Y.shape[0]:
        print(f"X shape before preprocessing: {X.shape}, Y shape: {Y.shape}")
        raise ValueError(f"Feature dimension mismatch: X has {X.shape[0]} features, Y has {Y.shape[0]} features")

    # Preprocess to (n_frames, n_features)
    X = np.swapaxes(X, -1, 0)
    Y = np.swapaxes(Y, -1, 0)
    X = X.reshape((X.shape[0], -1), order="F")
    Y = Y.reshape((Y.shape[0], -1), order="F")

    C = cdist(X, Y, metric='euclidean')
    is_transposed = X.shape[0] > Y.shape[0]
    if is_transposed:
        C = C.T

    max_0 = step_sizes[:, 0].max()
    max_1 = step_sizes[:, 1].max()
    D = np.ones((C.shape[0] + max_0, C.shape[1] + max_1)) * np.inf
    D[max_0, max_1:] = C[0, :]
    steps = np.zeros_like(D, dtype=np.int32)
    steps[0, :] = 1
    steps[:, 0] = 2

    D, steps = accumulate_cost(C, D, steps, step_sizes, weights_mul, weights_add, max_0, max_1)
    D = D[max_0:, max_1:]
    steps = steps[max_0:, max_1:]

    if np.all(np.isinf(D[-1])):
        raise ValueError("No valid subsequence path")
    start = np.argmin(D[-1, :])
    path = backtrack_path(steps, step_sizes, start)
    wp = np.asarray(path, dtype=int)
    if is_transposed or C.shape[0] > C.shape[1]:
        wp = np.fliplr(wp)

    return D, wp