from math import pi

CELERITAS = 2.998e10  # cm s-1
PLANCK = 6.62607004e-30  # kg cm^2 s-1
NA = 6.02214076e23  # mol-1
EH_2_PCM = 219474.63136320  # cm-1 Eh-1

DISTANCE = {'bohr': 5.29177210904e-9,
            'm': 100,
            'angstrom': 1e-8,
            'cm': 1}
ENERGY = {'Eh': 219474.63136320, 'kcal mol-1': 349.7550112241469, 'kJ mol-1': 83.59345392546533}
MASS = {'amu': 1 / NA / 1000}

K = 2 * pi * (CELERITAS / PLANCK) ** 0.5  # cm-1/2 kg-1/2
I_TO_B = PLANCK / (8 * pi * pi * CELERITAS)  # ??


def levi_civita(dtype=float):
    from numpy import zeros
    e_tensor = zeros([3, 3, 3], dtype=dtype)
    for i in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        e_tensor.itemset(i, 1)
        e_tensor.itemset(i[::-1], -1)
    e_tensor.setflags(write=False)
    return e_tensor


LEVI_CIVITA = levi_civita(int)
