import numpy as np
from scipy.linalg import inv, sqrtm

from .constants import LEVI_CIVITA


def centre_of_mass(pos, masses):
    return np.einsum('i, ia', masses, pos) / masses.sum()


def get_inertia_tensor(pos, masses):
    pos0 = pos - centre_of_mass(pos, masses)
    k = np.einsum('i, ia, ib -> ab', masses, pos0, pos0)
    return np.einsum('age, bde, gd -> ab', LEVI_CIVITA, LEVI_CIVITA, k)


def inertia_derivative(pos, masses, evecs):
    return 2 * np.einsum('age, bde, i, ig, idk -> kab', LEVI_CIVITA, LEVI_CIVITA, masses ** 0.5, pos, evecs)


def get_translation_vectors(pos, masses):
    pos0 = pos - centre_of_mass(pos, masses)
    return np.einsum('i,ia->a', masses, pos - pos0) / masses.sum() ** 0.5


def get_rotation_vectors(pos, masses):
    pos0 = pos - centre_of_mass(pos, masses)
    i_tensor = get_inertia_tensor(pos, masses)
    i = inv(sqrtm(i_tensor))
    m = masses ** 0.5
    return np.einsum('ab,i,bgd,ig,i,id->a', i, m, LEVI_CIVITA, pos0, m, pos - pos0)


def constrain(pos, masses):
    """
    align principal axes of inertia with coordinate axes &
    remove centre of mass
    """
    i_tensor = get_inertia_tensor(pos, masses)
    _, evecs = np.linalg.eigh(i_tensor)
    com = centre_of_mass(pos, masses)
    return np.einsum('ia,ab->ib', pos - com, evecs)


def get_coriolis_tensor(evecs):  # unitless
    shift = 0
    evecs = evecs[:, :, shift:]
    return np.einsum('abg, ibn, igm -> anm', LEVI_CIVITA, evecs, evecs)


def relations(pos, masses, evecs, linear=False):  # sanity checks
    shift = 6 - linear
    ndof = pos.size - shift
    evecs = evecs[:, :, shift:]

    m = masses ** 0.5
    test_id = np.einsum('ian,iam->nm', evecs, evecs)
    test_mass = abs(np.einsum('i,ian->n', m, evecs)).max()
    test_coords = abs(np.einsum('abg, i, ib, ign -> an', LEVI_CIVITA, m, pos, evecs)).max()
    print(f'test_id: {abs(test_id - np.eye(ndof)).max()}\n'
          f'test_mass: {test_mass}\n'
          f'test_coords: {test_coords}')
