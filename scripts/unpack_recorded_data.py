"""Verify and optionally unpack exact original experiment files. No credentials needed."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def entries():
    index = ROOT/'recorded_data/index.json'
    if not index.is_file():
        raise FileNotFoundError(
            'Original measurements are distributed separately. Extract '
            'hardware-aware-transport-qaoa-data.zip into this repository root '
            'to create recorded_data/. See the optional-data section in README.md.'
        )
    return json.loads(index.read_text())['files']

def safe_path(root, relative):
    root = root.resolve()
    path = (root/relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Unsafe archive path: {relative}')
    return path

def read_record(relative):
    record = next(r for r in entries() if r['path'] == relative)
    packed = safe_path(ROOT, record['archive']).read_bytes()
    if digest(packed) != record['archive_sha256']:
        raise ValueError(f'Archive checksum mismatch: {relative}')
    data = gzip.decompress(packed)
    if len(data) != record['bytes'] or digest(data) != record['sha256']:
        raise ValueError(f'Original checksum mismatch: {relative}')
    return data

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--destination', type=Path, default=ROOT)
    p.add_argument('--study', action='append', help='Only this outputs/<study> subtree. Repeatable.')
    args = p.parse_args()
    total = count = 0
    for r in entries():
        if args.study and not any(r['path'].startswith('outputs/'+s+'/') for s in args.study):
            continue
        packed = safe_path(ROOT, r['archive']).read_bytes()
        if digest(packed) != r['archive_sha256']:
            raise ValueError(f"Archive checksum mismatch: {r['path']}")
        data = gzip.decompress(packed)
        if len(data) != r['bytes'] or digest(data) != r['sha256']:
            raise ValueError(f"Original checksum mismatch: {r['path']}")
        if not args.verify_only:
            path = safe_path(args.destination, r['path'])
            if path.exists():
                if digest(path.read_bytes()) != r['sha256']:
                    raise FileExistsError(f'Refusing to overwrite changed file: {path}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        count += 1
        total += len(data)
    print(json.dumps(dict(files_verified=count, original_bytes=total,
                         unpacked=not args.verify_only), indent=2))

if __name__ == '__main__':
    main()
