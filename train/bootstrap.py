"""Start independent trainers from one immutable, checksummed checkpoint."""
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _experiment_files(directory, experiment, names):
    for description in Path(directory).glob("*/run.json"):
        try:
            run = json.loads(description.read_text())
        except (OSError, ValueError):
            continue
        if run.get("meta", {}).get("experimentId") != experiment:
            continue
        for name in names:
            path = description.parent / name
            if path.is_file():
                yield path


def experiment_checkpoint(directory, experiment):
    """The experiment's most recent weights: its latest policy.pt or best.pt.

    A restart carries on from where training got to, not from the champion.
    Taking best.pt used to throw away everything learned since the last
    promotion, which on a run that had not promoted yet was the whole run.
    """
    found = list(_experiment_files(directory, experiment, ("best.pt", "policy.pt")))
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def experiment_champion(directory, experiment):
    """The experiment's most recent best.pt: the policy selection has to beat."""
    found = list(_experiment_files(directory, experiment, ("best.pt",)))
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_seed(url, directory, experiment, host=None, timeout=300):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment):
        raise ValueError("experiment ID must contain only letters, digits, _ or -")
    if not url.startswith(("http://", "https://")):
        raise ValueError("bootstrap URL must use HTTP or HTTPS")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"seed-{experiment}.pt"
    request = urllib.request.Request(url, headers={"Host": host} if host else {})
    deadline = time.monotonic() + timeout
    while True:
        temporary = None
        try:
            with urllib.request.urlopen(request, timeout=min(20, max(0.1, timeout))) as response:
                expected = response.headers.get("X-Checkpoint-SHA256", "")
                if not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise ValueError("seed response has no SHA-256")
                digest = hashlib.sha256()
                size = 0
                with tempfile.NamedTemporaryFile(dir=folder, prefix=".seed-", delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 256 * 1024 * 1024:
                            raise ValueError("seed checkpoint exceeds 256 MiB")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size == 0 or digest.hexdigest() != expected:
                    raise ValueError("seed checkpoint checksum mismatch")
                temporary.replace(target)
                print(f"bootstrap | experiment {experiment} | sha256 {expected}", flush=True)
                return target
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if time.monotonic() >= deadline:
                raise RuntimeError("bootstrap checkpoint is unavailable") from error
            print(f"bootstrap waiting: {type(error).__name__}", flush=True)
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
