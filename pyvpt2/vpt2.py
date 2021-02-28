from functools import partial
from itertools import combinations_with_replacement

import numpy as np

from pyvpt2 import config
from pyvpt2.diffhess import diff_hess
from pyvpt2.rotation import get_inertia_tensor, get_coriolis_tensor, constrain
from pyvpt2.stencil import build_stencils_up_to_order, compute_derivatives, five_point_stencil
from pyvpt2.transform import transform_f2, transform_f3_Q2q, transform_f4_Q2q, inertia_to_b, transformed_energy_func, \
    transformed_hessian_func, transform_x2q, transform_x2Q, transform_f3_x2Q, transform_f4_x2Q
from pyvpt2.vibration import get_anharmonicity_matrix_hdcpt2, get_alpha_matrix
from pyvpt2.vibration import get_info


def relevant_fourth_derivative(derivative):
    return (*map(derivative.count, derivative),) in ((4, 4, 4, 4), (2, 2, 2, 2))


def check_potential_q(func, pos, q0, masses, evecs, omega, phi3, phi4, **units):
    energy = func(pos.flatten())
    q = transform_x2q(pos, masses, evecs, omega, **units) - q0
    estimate = (
            1 / 2 * np.einsum('u,u,u->', omega, q, q)
            + 1 / 6 * np.einsum('uvw,u,v,w->', phi3, q, q, q)
            + 1 / 24 * np.einsum('uvwx,u,v,w,x->', phi4, q, q, q, q)
    )
    return energy, estimate


def check_potential_big_q(func, pos, q0, masses, evecs, lamda, f3, f4):
    energy = func(pos.flatten())
    q = transform_x2Q(pos, masses, evecs) - q0
    estimate = (
            1 / 2 * np.einsum('u,u,u->', lamda, q, q)
            + 1 / 6 * np.einsum('uvw,u,v,w->', f3, q, q, q)
            + 1 / 24 * np.einsum('uvwx,u,v,w,x->', f4, q, q, q, q)
    )
    return energy, estimate


def get_derivatives_hessian(hess_func, pos, masses, h, name, in_q=True, linear=False, load=False, save=False, **units):
    root = f'{config.DERIVATIVES_DIR}/{name}'

    if load:
        omega = np.load(f'{root}.omega.npy')
        evecs = np.load(f'{root}.evecs.npy')
        phi3 = np.load(f'{root}.phi3.npy')
        phi4_small = np.load(f'{root}.phi4.npy')
        phi4 = np.zeros(4 * (omega.size,), dtype=float)
        for i, j in combinations_with_replacement(range(omega.size), 2):
            phi4[i, i, j, j] = phi4_small[i, j]
            phi4[j, j, i, i] = phi4_small[i, j]

    else:
        f2 = hess_func(pos, opt=True).reshape(*pos.shape, *pos.shape)
        lamda, omega, evecs = transform_f2(f2, masses, pos, linear, **units)

        if save:
            np.save(f'{root}.omega.npy', omega)
            np.save(f'{root}.evecs.npy', evecs)

        if in_q:
            big_q = transform_x2Q(pos, masses, evecs)
            qfunc = partial(transformed_hessian_func, func=hess_func, masses=masses, evecs=evecs)
            f3, f4 = diff_hess(big_q, qfunc, h)

        else:
            f3x, f4x = diff_hess(pos, hess_func, h)
            f3x = f3x.reshape(3 * [*pos.shape])
            f4x = f4x.reshape(4 * [*pos.shape])
            f3 = transform_f3_x2Q(f3x, masses, evecs)
            f4 = transform_f4_x2Q(f4x, masses, evecs)

        phi3 = transform_f3_Q2q(f3, omega, **units)
        phi4 = transform_f4_Q2q(f4, omega, **units)

        phi4_small = np.zeros(2 * (omega.size,), dtype=float)

        for i, j in combinations_with_replacement(range(omega.size), 2):
            phi4_small[i, j] = phi4[i, i, j, j]
            phi4_small[j, i] = phi4[i, i, j, j]

        if save:
            np.save(f'{root}.phi3.npy', phi3)
            np.save(f'{root}.phi4.npy', phi4_small)

    return omega, evecs, phi3, phi4


def get_derivatives(func, pos, masses, h, stencils, name, in_q=True, linear=False, load=False, save=False, **units):
    root = f'{config.DERIVATIVES_DIR}/{name}'

    if load:
        omega = np.load(f'{root}.omega.npy')
        evecs = np.load(f'{root}.evecs.npy')
        phi3 = np.load(f'{root}.phi3.npy')
        phi4_small = np.load(f'{root}.phi4.npy')
        phi4 = np.zeros(4 * (omega.size,), dtype=float)
        for i, j in combinations_with_replacement(range(omega.size), 2):
            phi4[i, i, j, j] = phi4_small[i, j]
            phi4[j, j, i, i] = phi4_small[i, j]

    else:
        f2 = compute_derivatives(func, pos.flatten(), h, stencils, order=2)

        f2 = f2.reshape(*pos.shape, *pos.shape)
        lamda, omega, evecs = transform_f2(f2, masses, pos, linear, **units)

        if save:
            np.save(f'{root}.omega.npy', omega)
            np.save(f'{root}.evecs.npy', evecs)

        if in_q:
            big_q = transform_x2Q(pos, masses, evecs)
            qfunc = partial(transformed_energy_func, func=func, masses=masses, evecs=evecs)
            f3 = compute_derivatives(qfunc, big_q, h * 2, stencils, order=3)
            f4 = compute_derivatives(qfunc, big_q, h * 2, stencils, order=4, filter_func=relevant_fourth_derivative)

        else:
            f3x = compute_derivatives(func, pos.flatten(), h * 2, stencils, order=3).reshape(3 * [*pos.shape])
            f4x = compute_derivatives(func, pos.flatten(), h * 2, stencils, order=4).reshape(4 * [*pos.shape])
            f3 = transform_f3_x2Q(f3x, masses, evecs)
            f4 = transform_f4_x2Q(f4x, masses, evecs)

        phi3 = transform_f3_Q2q(f3, omega, **units)
        phi4 = transform_f4_Q2q(f4, omega, **units)

        phi4_small = np.zeros(2 * (omega.size,), dtype=float)

        for i, j in combinations_with_replacement(range(omega.size), 2):
            phi4_small[i, j] = phi4[i, i, j, j]
            phi4_small[j, i] = phi4[i, i, j, j]

        if save:
            np.save(f'{root}.phi3.npy', phi3)
            np.save(f'{root}.phi4.npy', phi4_small)

    return omega, evecs, phi3, phi4


def vpt2(func, pos, masses, units, name, h=4.25e-3, in_q=True, linear=False, load=False, save=False, beta=5e5,
         hess=False):
    pos = constrain(pos.reshape(-1, 3), masses)

    if not hess:
        stencils = build_stencils_up_to_order(order=4, base_stencil=five_point_stencil())
        omega, evecs, phi3, phi4 = get_derivatives(func, pos, masses, h, stencils,
                                                   name, in_q, linear, load, save, **units)
    else:
        omega, evecs, phi3, phi4 = get_derivatives_hessian(func, pos, masses, h,
                                                           name, in_q, linear, load, save, **units)

    it = np.diagonal(get_inertia_tensor(pos, masses))
    b = inertia_to_b(it, **units)
    coriolis = get_coriolis_tensor(evecs)

    xij = get_anharmonicity_matrix_hdcpt2(omega, phi3, phi4, coriolis, b, beta=beta)
    alpha = get_alpha_matrix(omega, phi3, b, pos, masses, evecs, coriolis, **units)
    out_str = get_info(omega, xij.sum(0), alpha, rot=b, gauss=True)
    return out_str
