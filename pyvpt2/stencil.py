from collections import defaultdict
from itertools import combinations_with_replacement, product, permutations

import numpy as np


def rephrase(derivative):
    index_set = set(derivative)
    sorted_indices = tuple(sorted(index_set, key=derivative.count, reverse=True))
    return sorted_indices, (len(derivative), *map(derivative.count, sorted_indices),)


def two_point_stencil():
    return list(zip((-1, 1), (-1 / 2, 1 / 2)))


def five_point_stencil():
    return list(zip((-2, -1, 1, 2), (1 / 12, -8 / 12, 8 / 12, -1 / 12)))


def unique_indices(max_order, this=0, count=0, max_count=None):
    if max_count is None:
        max_count = max_order
    if count > max_count:
        return
    if max_order == 0:
        return
    yield (this,)
    for ii in unique_indices(max_order - 1, this=this, count=count + 1, max_count=max_count):
        yield (this, *ii)
    for ii in unique_indices(max_order - 1, this=this + 1, count=0, max_count=count):
        yield (this, *ii)


def build_stencil(derivative, base_stencil):
    order = len(derivative)
    unique = len(set(derivative))
    stencil = defaultdict(int)
    for steps in product(base_stencil, repeat=order):
        step_list = unique * [0]
        total_coefficient = 1
        for coordinate, (step, coefficient) in zip(derivative, steps):
            step_list[coordinate] += step
            total_coefficient *= coefficient
        stencil[tuple(step_list)] += total_coefficient
    return stencil


def dict_type():
    return defaultdict(int)


def build_stencils_up_to_order(order, base_stencil):
    derivatives = sorted(unique_indices(order), key=len)
    stencils = defaultdict(dict_type)
    for derivative in derivatives:
        _, label = rephrase(derivative)
        stencils[label] = build_stencil(derivative, base_stencil)
    return stencils


def compute_derivatives(f, x, h, stencils, order=1, filter_func=None):
    values = np.zeros(order * [x.size], dtype=float)
    derivatives = combinations_with_replacement(range(x.size), order)
    if filter_func is not None:
        derivatives = filter(filter_func, derivatives)
    derivatives = list(derivatives)

    for derivative in derivatives:
        print(derivative)
        _, value = compute_derivative(derivative, stencils=stencils, f=f, x=x, h=h)
        for perm in set(permutations(derivative)):
            values.itemset(perm, value)

    return values


def compute_derivative(derivative, stencils, f, x, h):
    indices, label = rephrase(derivative)
    stencil = stencils[label]
    x_ = x.copy()
    value = 0.0
    for steps, coefficient in stencil.items():
        x_[:] = x[:]
        for index, step in zip(indices, steps):
            x_[index] += step * h
        value += f(x_) * coefficient
    return derivative, value / h ** len(derivative)
