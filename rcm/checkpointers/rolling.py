"""Local rolling checkpoints: commit a new complete snapshot before retiring the old one."""
import os
from pathlib import Path
import re
import shutil


def checkpoint_name(iteration, archive_interval, rolling_interval=0):
    if iteration < 0 or archive_interval <= 0 or rolling_interval < 0:
        raise ValueError('Invalid checkpoint iteration/interval')
    prefix = 'iter' if not rolling_interval or iteration % archive_interval == 0 else 'latest_iter'
    return f'{prefix}_{iteration:09}'


def publish_and_prune(directory, checkpoint_file):
    root = Path(directory)
    if not (root / checkpoint_file).is_dir():
        raise FileNotFoundError(root / checkpoint_file)
    # This function is invoked only after every DCP component has saved successfully.
    temporary = root / f'.latest_checkpoint.{os.getpid()}.tmp'
    with temporary.open('w') as f:
        f.write(checkpoint_file + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, root / 'latest_checkpoint.txt')
    for old in root.iterdir():
        if (old.name != checkpoint_file and re.fullmatch(r'latest_iter_\d{9,}', old.name)
                and old.is_dir() and not old.is_symlink()):
            shutil.rmtree(old)
