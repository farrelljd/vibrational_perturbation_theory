import subprocess
from io import StringIO

import numpy as np

from pyorca import orcaconfig as config


class OrcaWrapper:
    def __init__(self, jobname, elements, theory, basis, *args, parallel=None):
        self.elements = elements
        self.jobname = jobname
        if theory.lower() == 'mp2':
            args = (*args, 'nofrozencore')
        self.header = f'! {" ".join([theory, basis, *args])} bohrs tightscf ' + '{job_type:}\n'
        if parallel is not None:
            self.header += f'%pal nprocs {parallel}\n    end\n'

    def run_orca(self, input_str):
        with open(f'{self.jobname}.inp', 'w') as f:
            f.write(input_str)
        args = [config.ORCA_EXEC, f'{self.jobname}.inp']
        with subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            out = proc.stdout.read().decode()
            if '****ORCA TERMINATED NORMALLY****' not in out:
                raise RuntimeError(out)
        return

    def write_coord_string(self, coords, charge, multiplicity):
        coords_str = f'* xyz {charge} {multiplicity}\n'
        coords_str += '\n'.join(
            f'{e} {x:20.12f} {y:20.12f} {z:20.12f}' for e, (x, y, z) in zip(self.elements, coords.reshape(-1, 3)))
        coords_str += '\n*\n'
        return coords_str

    def run_job(self, job_type, coords, charge=0, multiplicity=1):
        input_str = self.header.format(job_type=job_type)
        input_str += self.write_coord_string(coords, charge, multiplicity)
        self.run_orca(input_str)

    def get_scf_energy(self, coords, charge=0, multiplicity=1):
        self.run_job('svp', coords, charge, multiplicity)
        properties = open(f'{self.jobname}_property.txt', 'r').readlines()
        scf = None
        for line in properties:
            if 'SCF Energy' in line:
                scf = float(line.strip().split()[-1])
        return scf

    def optimize(self, coords, charge=0, multiplicity=1):
        self.run_job('verytightopt', coords, charge, multiplicity)
        return np.genfromtxt(f'{self.jobname}.xyz', skip_header=2, usecols=(1, 2, 3)) / 5.29177210904e-1

    def generate_hessian_columns(self, size):
        hess_lines = open(f'{self.jobname}.hess', 'r').readlines()
        start = hess_lines.index('$hessian\n')
        end = hess_lines.index('$vibrational_frequencies\n')
        hess_lines = hess_lines[start + 2:end - 1]
        while hess_lines:
            block, hess_lines = hess_lines[:size + 1], hess_lines[size + 1:]
            block = "".join(block[1:])
            yield np.loadtxt(StringIO(block))[:, 1:]

    def get_hessian(self, coords, charge=0, multiplicity=1, opt=False):
        job_type = ('verytightopt ' if opt else '') + 'anfreq'
        n = coords.size // 3
        self.run_job(job_type, coords, charge, multiplicity)
        return np.hstack([*self.generate_hessian_columns(coords.size)]).reshape((n, 3, n, 3,))


def main():
    elements = 'O H H'.split()
    pot = OrcaWrapper(jobname='job', elements=elements, theory='MP2', basis='6-31G*')
    coords = np.array([0.00000000000000, 0.00000000000000, 0.05568551114552,
                       0.00000000000000, 0.76411921207143, -0.54015925557276,
                       0.00000000000000, -0.76411921207143, -0.64015925557275]) / 5.29177210904e-1
    coords = pot.optimize(coords)
    v = pot.get_scf_energy(coords)
    print(v)


if __name__ == '__main__':
    main()
