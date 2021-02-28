from itertools import product

import numpy as np


def diff_hess(coords, hess_func, h=5.0e-3):
    d = coords.size
    coords_copy = coords.copy()
    forward = np.zeros([d, d, d], dtype=float)
    backward = np.zeros([d, d, d], dtype=float)
    hess = hess_func(coords)
    third_derivatives = np.zeros([d, d, d], dtype=float)
    fourth_derivatives = np.zeros([d, d, d, d], dtype=float)
    for i, (forward_matrix, backward_matrix) in enumerate(zip(forward, backward)):
        print(f'differentiating hessian matrix w.r.t normal coordinate {i}')
        coords[i] += h
        forward_matrix[:] = hess_func(coords)
        coords[:] = coords_copy[:]

        coords[i] -= h
        backward_matrix[:] = hess_func(coords)
        coords[:] = coords_copy[:]

    first = (forward - backward)
    second = (forward + backward)

    for i, j, k in product(range(d), repeat=3):
        third_derivatives[i, j, k] = 1 / 3 * (first[i, j, k] + first[j, k, i] + first[k, i, j]) / 2 / h
    for i, j in product(range(d), repeat=2):
        fourth_derivatives[i, i, j, j] = (second[j, i, i] - 2 * hess[i, i] + second[i, j, j] - 2 * hess[
            j, j]) / h / h / 2
    return third_derivatives, fourth_derivatives
