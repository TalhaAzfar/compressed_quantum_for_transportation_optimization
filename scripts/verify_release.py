"""Verify distributed source, configuration, compact data, and measurement archives."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows = (ROOT/'SHA256SUMS').read_text().splitlines()
    for row in rows:
        expected, relative = row.split('  ', 1)
        path = (ROOT/relative).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError(f'Unsafe manifest path: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
    print(f'Verified {len(rows)} distributed files.')

if __name__ == '__main__':
    main()
