import argparse
import json
from pathlib import Path


FAMILIES = {
    "Pick": "pick_and_place_simple",
    "Look": "look_at_obj_in_light",
    "Clean": "pick_clean_then_place_in_recep",
    "Heat": "pick_heat_then_place_in_recep",
    "Cool": "pick_cool_then_place_in_recep",
    "Pick2": "pick_two_obj_and_place",
}
EXPECTED_TOTALS = {
    "train": 3553,
    "valid_train": 200,
    "valid_seen": 140,
    "valid_unseen": 134,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=Path, help="DuetHarness processed/games.jsonl")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    aliases = {task_type: name for name, task_type in FAMILIES.items()}
    splits = {split: {name: [] for name in FAMILIES} for split in EXPECTED_TOTALS}
    present = set()
    with args.games.open() as games:
        for line in games:
            game = json.loads(line)
            split = game["official_split"]
            path = game["game_file"]
            if path in present or not (args.games.parent.parent / path).is_file():
                raise ValueError(f"missing or duplicate game file: {path}")
            present.add(path)
            splits[split][aliases[game["task_type"]]].append(path)

    for split, families in splits.items():
        counts = {name: len(paths) for name, paths in families.items()}
        if sum(counts.values()) != EXPECTED_TOTALS[split] or not all(counts.values()):
            raise ValueError(f"unexpected family counts for {split}: {counts}")
        print(split, counts)

    args.output.write_text(
        json.dumps({"stage_order": list(FAMILIES), "splits": splits}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
