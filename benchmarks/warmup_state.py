"""Check the original trainer's committed step and pin the shared branch point."""
import errno
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


def committed_step(root, ranks):
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        if list(root.glob("global_step_*")):
            raise RuntimeError("checkpoint directories exist without a commit marker")
        return 0
    step = int(tracker.read_text().strip())
    if step <= 0:
        raise ValueError("invalid checkpoint step")
    directory = root / f"global_step_{step}"
    check_files(directory, ranks)
    return step


def check_files(directory, ranks):
    required = [directory / "data.pt"]
    for stem in ("model", "optim", "extra_state"):
        shards = [directory / "actor" / f"{stem}_world_size_{ranks}_rank_{rank}.pt" for rank in range(ranks)]
        if set(shards) != set((directory/"actor").glob(f"{stem}_world_size_*_rank_*.pt")):
            raise RuntimeError(f"unexpected {stem} shard set in {directory.name}")
        required.extend(shards)
    for path in required:
        # Reading just the zip directory detects missing/truncated files without
        # loading tensors. Full recovery still requires a successful trainer load.
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise ValueError(f"empty torch archive {path.name}")


def pin_shared(root, ranks):
    if committed_step(root, ranks) != 100:
        raise RuntimeError("the committed shared branch point must be step 100")
    return link_generation(root / "global_step_100", root.parent / "protected")


def link_generation(source, parent):
    parent.mkdir(exist_ok=True)

    target = parent / source.name
    if target.is_dir():
        if not os.path.samefile(target/"data.pt", source/"data.pt"):
            raise RuntimeError("different generation already present at destination")
        return target
    with tempfile.TemporaryDirectory(dir=parent) as stage:
        tree = Path(stage)/source.name
        shutil.copytree(source, tree, copy_function=os.link)
        try:
            tree.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not os.path.samefile(
                    target/"data.pt", source/"data.pt"):
                raise
    return target

def retained_dir(work):
    return work.with_name(work.name + ".retained")


def generations(parent):
    return sorted((int(p.name[12:]), p) for p in parent.glob("global_step_*")
                  if p.is_dir() and p.name[12:].isdigit())


def rotate_fallback(work, latest, ranks):
    """Before the next save, promote the last committed step to the fallback slot."""
    root = work/"ckpt"
    if committed_step(root, ranks) != latest:
        return
    archive = retained_dir(work)
    link_generation(root/f"global_step_{latest}", archive)
    # Only this warmup's superseded, numbered generations are eligible.
    # On resume the upstream retention list is empty, so retire native
    # leftovers too. The current native and its pinned fallback stay intact.
    for parent in (archive, root):
        for step, path in generations(parent):
            if step < latest:
                shutil.rmtree(path)


def resume_step(work, ranks):
    root = work/"ckpt"
    try:
        step = committed_step(root, ranks)
        if step or not generations(retained_dir(work)):
            return step
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile):
        if not generations(retained_dir(work)):
            raise
    # The old actor may have been unlinked before an interrupted new save.
    # Only restore a validated, previously committed fallback. Do not mix
    # shards from different writes. Preserve an incomplete target for diagnosis.
    step, previous = generations(retained_dir(work))[-1]
    check_files(previous, ranks)
    root.mkdir(exist_ok=True)
    target = root/previous.name
    if target.exists():
        with tempfile.NamedTemporaryFile(dir=root, prefix="replaced-", delete=True) as marker:
            quarantine = Path(marker.name + ".dir")
        target.rename(quarantine)
    link_generation(previous, root)
    marker = root/"latest_checkpointed_iteration.txt"
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(str(step))
    tmp.replace(marker)
    return step
