"""Use the same double-precision AQC settings as the experiment runner."""
import os

os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('QUIMB_NUMBA_CACHE', 'FALSE')
