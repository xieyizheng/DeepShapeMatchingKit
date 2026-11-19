"""Create the EchoMatch pair-level split from official PARTIALSMAL training data."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory


EXPECTED_PAIRS = 152


def create_split(source, destination, seed, val_pairs):
    source = source.resolve()
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination already exists: {destination}")
    if not 0 < val_pairs < EXPECTED_PAIRS:
        raise ValueError(f"Validation pair count must be between 1 and {EXPECTED_PAIRS - 1}")

    maps_dir = source / "maps"
    shapes_dir = source / "shapes"
    if not maps_dir.is_dir() or not shapes_dir.is_dir():
        raise ValueError(f"Expected maps/ and shapes/ under {source}")

    pairs = defaultdict(dict)
    for map_file in sorted(maps_dir.glob("*.vts")):
        parts = map_file.stem.split("_")
        if len(parts) != 8 or not all(parts) or not map_file.is_file():
            raise ValueError(f"Invalid directional map: {map_file}")
        first, second = "_".join(parts[:4]), "_".join(parts[4:])
        if first == second:
            raise ValueError(f"Expected two distinct shapes in {map_file.name}")
        pairs[tuple(sorted((first, second)))][(first, second)] = map_file

    for (first, second), directions in pairs.items():
        if directions.keys() != {(first, second), (second, first)}:
            raise ValueError(f"Missing reverse map for {first} and {second}")
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} training pairs, found {len(pairs)}")

    for shape in sorted({shape for pair in pairs for shape in pair}):
        shape_file = shapes_dir / f"{shape}.off"
        if not shape_file.is_file():
            raise ValueError(f"Missing shape: {shape_file}")

    pair_keys = sorted(pairs)
    random.Random(seed).shuffle(pair_keys)
    partitions = {"train": pair_keys[val_pairs:], "val": pair_keys[:val_pairs]}
    manifest = {"seed": seed, "splits": {}}

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".partialsmal-", dir=destination.parent) as staging:
        staged_split = Path(staging) / "split"
        for name, keys in partitions.items():
            root = staged_split / name
            (root / "maps").mkdir(parents=True)
            (root / "shapes").symlink_to(shapes_dir, target_is_directory=True)
            selected_maps = sorted(
                map_file for key in keys for map_file in pairs[key].values()
            )
            for map_file in selected_maps:
                (root / "maps" / map_file.name).symlink_to(map_file)
            manifest["splits"][name] = [map_file.name for map_file in selected_maps]

        (staged_split / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Destination already exists: {destination}")
        staged_split.rename(destination)

    for name, keys in partitions.items():
        print(f"{name}: {len(keys)} pairs, {len(manifest['splits'][name])} directional maps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official PARTIALSMAL training root")
    parser.add_argument("--destination", type=Path, required=True, help="New split directory")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-pairs", type=int, default=16)
    args = parser.parse_args()
    try:
        create_split(args.source, args.destination, args.seed, args.val_pairs)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
