import numpy as np

from .constants import K, MASS, DISTANCE
from .rotation import inertia_derivative, get_inertia_tensor


def delta_ijk(i, j, k):
    return (i + j - k) * (i + j + k) * (i - j + k) * (i - j - k)


def resonance_term(wi, wj, wk, k2, alpha, beta):
    delta = (wi + wj - wk)
    possibly_resonant = k2 / delta
    eps = abs(delta) / 2
    degeneracy_corrected = np.sign(delta) * ((eps * eps + k2) ** 0.5 - eps)
    test = (eps * eps * k2) ** 0.5
    lamda = (np.tanh(alpha * (test - beta)) + 1) / 2
    return lamda * possibly_resonant + (1 - lamda) * degeneracy_corrected


def get_anharmonicity_matrix_hdcpt2(omega, phi3, phi4, coriolis, b, alpha=1e0, beta=5e5):
    size = omega.size
    xij = np.zeros((3, size, size), dtype=float)

    for i in range(size):
        wi = omega[i]
        for j in range(i, size):
            wj = omega[j]
            if i == j:
                xij[2, i, i] += phi4[i, i, i, i] / 16
                for k in range(size):
                    wk = omega[k]
                    k2 = phi3[i, i, k] ** 2 / 32
                    non_resonant = k2 * (1 / (2 * wi + wk) + 4 / wk)
                    rii_k = resonance_term(wi, wi, wk, k2, alpha, beta)
                    xij[1, i, i] -= non_resonant - rii_k
            else:
                xij[2, i, j] += phi4[i, i, j, j] / 4
                for k in range(size):
                    wk = omega[k]
                    xij[1, i, j] -= phi3[i, i, k] * phi3[j, j, k] / wk / 4

                    k2 = phi3[i, j, k] ** 2 / 8
                    non_resonant = k2 / (wi + wj + wk)
                    rij_k = resonance_term(wi, wj, wk, k2, alpha, beta)
                    rik_j = resonance_term(wi, wk, wj, k2, alpha, beta)
                    rjk_i = resonance_term(wj, wk, wi, k2, alpha, beta)
                    xij[1, i, j] -= non_resonant + rik_j + rjk_i - rij_k
            xij[0, i, j] += (wi ** 2 + wj ** 2) / wi / wj * (b[:] * coriolis[:, i, j] ** 2).sum()
            xij[:, j, i] = xij[:, i, j]
    return xij


def get_alpha_matrix(omega, phi3, rot, pos, masses, evecs, cor, distance, mass, energy):
    a = inertia_derivative(pos, masses, evecs)
    a_scaled = a * MASS[mass] ** 0.5 * DISTANCE[distance]
    inertia = np.diagonal(get_inertia_tensor(pos, masses))
    alpha = np.zeros([omega.size, 3], dtype=float)

    inertia_term = np.zeros_like(alpha)
    coriolis_term = np.zeros_like(alpha)
    anharmonic_term = np.zeros_like(alpha)
    summed_coriolis_term = np.zeros_like(rot)
    rb_factor = np.zeros_like(alpha)
    for r in range(omega.size):
        for b in range(3):
            rb_factor[r, b] = -2 * rot[b] ** 2 / omega[r]
            for z in range(3):
                inertia_term[r, b] += 3 * a[r, b, z] ** 2 / 4 / inertia[z]
            for s in range(omega.size):
                if s == r:
                    continue
                coriolis_term[r, b] += cor[b, r, s] ** 2 * (3 * omega[r] ** 2 + omega[s] ** 2) / (
                        omega[r] ** 2 - omega[s] ** 2)
                if s <= r:
                    continue
                summed_coriolis_term[b] += 2 * rot[b] ** 2 * cor[b, r, s] * (omega[r] - omega[s]) ** 2 / (
                        omega[r] * omega[s] * (omega[r] + omega[s]))
            for s in range(omega.size):
                anharmonic_term[r, b] += K / 2 * phi3[r, r, s] * a_scaled[s, b, b] * (
                        omega[r] / omega[s] ** (3 / 2))
    alpha = rb_factor * (inertia_term + coriolis_term + anharmonic_term)
    return alpha


def get_fundamental(mode, omega, xij):
    size = omega.size
    return omega[mode] + 2 * xij[mode, mode] + 0.5 * sum(xij[mode, j] for j in range(size) if j != mode)


def get_overtone(mode, omega, xij):
    return 2 * get_fundamental(mode, omega, xij) + 2 * xij[mode, mode]


def get_combination(mode1, mode2, omega, xij):
    return get_fundamental(mode1, omega, xij) + get_fundamental(mode2, omega, xij) + xij[mode1, mode2]


def get_info(omega, xij, alpha, rot, gauss=False):
    info = []
    modes = range(omega.size)
    rot = rot - 0.5 * alpha.sum(0)
    for mode in modes:
        omega_ = omega[mode]
        fundamental = get_fundamental(mode, omega, xij)
        a, b, c = alpha[mode]
        if a > 1e10:
            a = np.inf
        info.append(f'{omega_:18.3f}{fundamental:18.3f}{a:18.6f}{b:18.6f}{c:18.6f}')
    info.append('{:36}{:18.6f}{:18.6f}{:18.6f}'.format('', *rot))
    return '\n'.join(info[::-1] if gauss else info)
