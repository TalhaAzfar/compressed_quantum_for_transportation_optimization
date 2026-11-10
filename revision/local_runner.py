"""Run a revision module with an optional local dependency directory.

An ordinary environment containing the required packages needs no --extra-site.
The extra directory is appended, so it only fills missing packages rather than
replacing packages already present in the selected Python environment.
"""
import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-site")
    parser.add_argument("module")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.extra_site:
        if not os.path.isdir(args.extra_site):
            parser.error("--extra-site must be an existing directory")
        sys.path.append(args.extra_site)
    os.environ.setdefault("QUIMB_NUMBA_CACHE", "FALSE")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aqc_revision_matplotlib")
    sys.argv = [args.module, *args.args]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
