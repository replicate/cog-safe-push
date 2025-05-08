import re
import subprocess

from replicate.model import Model

from . import log


def push(model: Model, dockerfile: str | None, fast_push: bool = False) -> str:
    url = f"r8.im/{model.owner}/{model.name}"
    log.info(f"Pushing to {url}")
    cmd = ["cog", "push", url]
    if dockerfile:
        cmd += ["--dockerfile", dockerfile]
    if fast_push:
        cmd += ["--x-fast"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    sha256_id = None
    assert process.stdout
    for line in process.stdout:
        log.v(line.rstrip())  # Print output in real-time
        if "latest: digest: sha256:" in line:
            match = re.search(r"sha256:([a-f0-9]{64})", line)
            if match:
                sha256_id = match.group(1)
        # In the case of fast push, we get the version from the identifier printed to stdout
        elif "New Version:" in line:
            potential_sha256_id = line.split(":")[-1]
            if bool(re.match(r"^[a-f0-9]{64}$", potential_sha256_id)):
                sha256_id = potential_sha256_id

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, ["cog", "push", url])

    if not sha256_id:
        raise ValueError("No sha256 ID found in cog push output")

    return sha256_id
