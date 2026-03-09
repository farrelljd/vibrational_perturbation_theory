# PyVPT2

PyVPT2 is a Python software package developed to apply vibrational second-order perturbation theory (VPT2) to arbitrary model chemistries. Specifically, the package implements hybrid degeneracy-corrected VPT2.

## Key Features

* PyVPT2 numerically computes third- and fourth-order derivatives from a user-provided function that calculates the energy, gradient, or Hessian at a point on a potential energy surface.
* The software computes anharmonicity-corrected vibrational frequencies.
* It calculates excited-state rotational constants coupled to vibrational modes.
* The package includes a minimal wrapper for ORCA, a free-of-charge quantum chemistry package, to leverage analytical second-derivatives at the MP2 level.

## Acknowledgments

The development of PyVPT2 was supported by the National Natural Science Foundation of China (NSFC). Funding was provided through the Research Fund for International Young Scientists under grant number 21850410459.
