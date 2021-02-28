import numpy as np

from .constants import K, I_TO_B, DISTANCE, MASS, ENERGY
from .rotation import relations


def transform_x2Q(x, masses, evecs):
    m = masses ** 0.5
    return np.einsum('iak, i, ia -> k', evecs, m, x)


def transform_Q2x(big_q, masses, evecs):
    w = 1 / masses ** 0.5
    return np.einsum('i, iak, k -> ia', w, evecs, big_q)


def transform_x2q(x, masses, evecs, omega, distance, mass, energy):
    conversion = MASS[mass] ** 0.5 * DISTANCE[distance] * K
    sqrt_omega = omega ** 0.5
    return conversion * sqrt_omega * transform_x2Q(x, masses, evecs)


def transform_q2x(q, masses, evecs, omega, distance, mass, energy):
    conversion = MASS[mass] ** 0.5 * DISTANCE[distance] * K
    sqrt_omega = omega ** 0.5
    return transform_Q2x(q / conversion / sqrt_omega, masses, evecs)


def transformed_energy_func(big_q, func, masses, evecs):
    x = transform_Q2x(big_q, masses, evecs)
    return func(x.flatten())


def transformed_hessian_func(big_q, func, masses, evecs):
    x = transform_Q2x(big_q, masses, evecs)
    hess = func(x).reshape(*x.shape, *x.shape)
    f2 = transform_f2_x2Q(hess, masses, evecs)
    return f2


def transform_f2(f2, masses, pos, linear, distance, mass, energy):
    m = masses ** 0.5
    w = 1 / m
    f2 = np.einsum('i, iajb, j -> iajb', w, f2, w)
    lamda, evecs = np.linalg.eigh(f2.reshape(pos.size, pos.size))

    evecs = evecs.reshape(*pos.shape, pos.size)
    relations(pos, masses, evecs, linear)

    trans_rot = 6 - linear
    evecs = evecs[:, :, trans_rot:]
    lamda = lamda[trans_rot:]
    conversion = ENERGY[energy] / (MASS[mass] ** 0.5 * DISTANCE[distance] * K) ** 2
    omega = (lamda * conversion) ** 0.5
    return lamda, omega, evecs


def transform_f2_x2Q(f2, masses, evecs):
    w = 1 / masses ** 0.5
    return np.einsum('i, iau, j, jbv, iajb -> uv', w, evecs, w, evecs, f2)


def transform_f3_x2Q(f3, masses, evecs):
    w = 1 / masses ** 0.5
    return np.einsum('i, iau, j, jbv, k, kgw, iajbkg -> uvw', w, evecs, w, evecs, w, evecs, f3)


def transform_f3_Q2q(f3, omega, distance, mass, energy):
    conversion = ENERGY[energy] / (MASS[mass] ** 0.5 * DISTANCE[distance] * K) ** 3
    sqrt_omega = 1 / omega ** 0.5
    phi3 = f3 * conversion
    return np.einsum('u,v,w,uvw->uvw', sqrt_omega, sqrt_omega, sqrt_omega, phi3)


def transform_f4_x2Q(f4, masses, evecs):
    w = 1 / masses ** 0.5
    return np.einsum('i, iau, j, jbv, k, kgw, l, ldx, iajbkgld -> uvwx', w, evecs, w, evecs, w, evecs, w, evecs, f4)


def transform_f4_Q2q(f4, omega, distance, mass, energy):
    conversion = ENERGY[energy] / (MASS[mass] ** 0.5 * DISTANCE[distance] * K) ** 4
    sqrt_omega = 1 / omega ** 0.5
    phi4 = f4 * conversion
    return np.einsum('u,v,w,x,uvwx->uvwx', sqrt_omega, sqrt_omega, sqrt_omega, sqrt_omega, phi4)


def inertia_to_b(inertia, distance, mass, energy):
    conversion = MASS[mass] * DISTANCE[distance] ** 2
    return I_TO_B / (inertia * conversion)


def transform_lambda(lamda, distance, mass, energy):
    conversion = ENERGY[energy] / (MASS[mass] ** 0.5 * DISTANCE[distance] * K) ** 2
    return (lamda * conversion) ** 0.5
