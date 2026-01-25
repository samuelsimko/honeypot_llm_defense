import subprocess
from typing import List, Optional
from .base import Backend


class SlurmBackend(Backend):
    def __init__(self, partition, account, gres):
        self.partition = partition
        self.account = account
        self.gres = gres

    def submit(
        self,
        *,
        name: str,
        command: List[str],
        time: str,
        output_log: str,
        error_log: str,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "-p", self.partition,
            "--account", self.account,
            "--gres", self.gres,
            "--time", time,
            "--job-name", name,
            "--output", output_log,
            "--error", error_log,
        ]

        if depends_on:
            sbatch_cmd.append(
                f"--dependency=afterok:{':'.join(depends_on)}"
            )

        wrapped = (
            "bash -lc '"
            "set -euo pipefail; "
            "cd ~/honeypot_llm_defense; "
            "source ~/venv/bin/activate; "
            "source .env; "
            "nvidia-smi;"
            "sleep 3;"
            + " ".join(command)
            + "'"
        )

        sbatch_cmd += ["--wrap", wrapped]

        print("▶ [SLURM]", " ".join(sbatch_cmd))
        job_id = subprocess.check_output(sbatch_cmd).decode().strip()
        print(f"  ↳ job_id={job_id}")
        return job_id
