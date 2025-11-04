import re
import subprocess

from . import log


def push(
    model_owner: str,
    model_name: str,
    dockerfile: str | None,
    fast_push: bool = False,
    use_cog_base_image: bool = True,
) -> str:
    url = f"r8.im/{model_owner}/{model_name}"
    log.info(f"Pushing to {url}")
    cmd = ["cog", "push", url]
    if dockerfile:
        cmd += ["--dockerfile", dockerfile]
    if fast_push:
        cmd += ["--x-fast"]
    if not use_cog_base_image:
        cmd += ["--use-cog-base-image=false"]
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
        raise subprocess.CalledProcessError(process.returncode, cmd)

    if not sha256_id:
        raise ValueError("No sha256 ID found in cog push output")

    return sha256_id
