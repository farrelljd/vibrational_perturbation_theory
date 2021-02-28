import os


class OrcaConfig:
    ORCA_EXEC = '/home/farrelljd/apps/orca/orca'

    def __init__(self):
        if 'LD_LIBRARY_PATH' not in os.environ:
            os.environ['LD_LIBRARY_PATH'] = ':/home/farrelljd/apps/orca'


class PyVPT2Config:
    DERIVATIVES_DIR = './data/derivative'
    FREQS_DIR = './data/freq'
    XYZS_DIR = './data/xyz'
    PROCS = 4

    def __init__(self):
        for directory in self.DERIVATIVES_DIR, self.FREQS_DIR, self.XYZS_DIR:
            os.makedirs(directory, exist_ok=True)
