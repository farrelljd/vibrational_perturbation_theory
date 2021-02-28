import numpy as np

from pyorca import OrcaWrapper
from pyvpt2 import vpt2

name = 'watertest'

coords = np.array([0.000000, 0.000000, 0.076451,
                   0.000000, 1.426155, -1.006363,
                   0.000000, -1.426155, -1.006363])  # water molecule
elements = 'O H H'.split()
masses = np.array([15.999, 1.008, 1.008])
units = {'distance': 'bohr', 'mass': 'amu', 'energy': 'Eh'}
h = 0.003
beta = 1e5

orca = OrcaWrapper(jobname='job', elements=elements, theory='RHF', basis='6-31G*', parallel=4)
coords = orca.optimize(coords)

hess_func = orca.get_hessian
output = vpt2(hess_func, coords, masses, units, name, h, save=True, beta=beta, hess=True)
print(output)
