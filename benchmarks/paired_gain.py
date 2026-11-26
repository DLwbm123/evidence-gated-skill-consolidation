"""Evaluate fixed game-paired step-150 predictions for one continuation seed."""
import argparse
import csv
import json
from pathlib import Path
from random import Random
from statistics import fmean


def paired_gain(manifest, results):
    splits = manifest["splits"]
    expected = {s: {g for games in splits[s].values() for g in games}
                for s in ("valid_seen", "valid_unseen")}
    if [len(games) for games in expected.values()] != [140, 134]:
        raise ValueError("unexpected evaluation game inventory")
    cells = {}
    for row in results:
        split, game, method = row["split"], row["game"], row["method"]
        repeat, won = int(row["repeat"]), int(row["won"])
        key = split, game, repeat, method
        if (game not in expected.get(split, ()) or method not in ("grpo", "pr_opd")
                or repeat not in range(4) or won not in (0, 1) or key in cells):
            raise ValueError("invalid or duplicate outcome")
        cells[key] = won
    rng = Random(0)
    report = {}
    for split, games in expected.items():
        gains = []
        for game in sorted(games):
            # Missing either arm/repeat is a data error, never a counted failure.
            gains.append(fmean(cells[split,game,r,"pr_opd"]-cells[split,game,r,"grpo"] for r in range(4)))
        draws = sorted(fmean(rng.choices(gains,k=len(gains))) for _ in range(10000))
        report[split] = dict(games=len(games), episodes_per_arm=4*len(games), gain=fmean(gains),
                             percentile_95=[draws[249], draws[9749]])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results", type=Path, help="CSV columns split,game,repeat,method,won")
    args = parser.parse_args()
    with args.results.open(newline="") as f:
        print(json.dumps(paired_gain(json.loads(args.manifest.read_text()), csv.DictReader(f)),indent=2))
