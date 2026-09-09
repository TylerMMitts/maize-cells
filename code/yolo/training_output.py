# Separates what a training run produces into weights and everything else.
#
# ultralytics writes checkpoints and figures into one folder, which leaves
# 85 MB weights sitting beside PR curves and validation previews. Weights are
# an input to later runs; curves and previews are a result you look at once.
# They belong in different places, so this moves the second kind into
# results/training/<model_name>/ and leaves only weights under models/.
#
# It also renames checkpoints to carry the model name, because a loose
# best.pt or epoch10.pt is unidentifiable the moment it is copied anywhere.

import hashlib
import re
import shutil
from pathlib import Path

from code.config import RESULTS_FOLDER

TRAINING_RESULTS = RESULTS_FOLDER / 'training'

# Files that are inputs to a later run and stay under models/.
WEIGHT_SUFFIXES = {'.pt', '.onnx', '.engine'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _move_verified(src, dst, overwrite=False):
    # Copy, compare checksums, then remove the original. A plain move would be
    # cheaper, but a truncated copy across a drive boundary is silent and these
    # are the only record of a run that took hours.
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    before = _sha(src)
    shutil.copy2(src, dst)
    if _sha(dst) != before:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f'checksum mismatch copying {src} -> {dst}')
    src.unlink()


def checkpoint_name(model_name, stem):
    # best.pt -> <model>_best.pt, epoch10.pt -> <model>_epoch_10.pt
    if stem.startswith(f'{model_name}_'):
        return None                       # already renamed; leave it alone
    m = re.fullmatch(r'epoch[_]?(\d+)', stem)
    if m:
        return f'{model_name}_epoch_{int(m.group(1))}.pt'
    if stem in ('best', 'last'):
        return f'{model_name}_{stem}.pt'
    return None


def split_run_output(run_dir, model_name, dry_run=False, rename_weights=True):
    # Move figures and logs out of a finished run, and name its checkpoints.
    #
    # Safe to call twice: files already moved or already renamed are skipped,
    # so re-running a training script does not fail on the previous run.
    #
    # rename_weights is turned off when migrating runs that finished before
    # this split existed. Their weights are what the pipeline currently loads,
    # and renaming a file the project is actively using buys nothing that
    # config's two-name lookup does not already handle.
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return {'moved': 0, 'renamed': 0, 'results_dir': None}

    dest = TRAINING_RESULTS / model_name
    moved = renamed = 0

    for path in sorted(run_dir.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)

        if path.suffix.lower() in WEIGHT_SUFFIXES:
            if not rename_weights:
                continue
            new = checkpoint_name(model_name, path.stem)
            if new and not (path.parent / new).exists():
                if not dry_run:
                    path.rename(path.parent / new)
                renamed += 1
            continue

        target = dest / rel
        # Overwrite rather than skip. Skipping looks safer but silently drops a
        # new run's figures whenever an older run has already written there,
        # which leaves the stale curves in place and the fresh ones stranded in
        # models/. Retraining is meant to supersede. Calling this twice on one
        # run is still safe because the first call removes the source.
        if not dry_run:
            _move_verified(path, target, overwrite=True)
        moved += 1

    # tidy away the now-empty folders the figures used to live in
    if not dry_run:
        for d in sorted((p for p in run_dir.rglob('*') if p.is_dir()), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

    return {'moved': moved, 'renamed': renamed, 'results_dir': dest}


def report(result, model_name):
    if result['results_dir'] is None:
        print(f'  {model_name}: nothing to split')
        return
    print(f'  {model_name}: {result["moved"]} files -> {result["results_dir"]}, '
          f'{result["renamed"]} checkpoints renamed')
