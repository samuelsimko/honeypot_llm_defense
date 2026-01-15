import subprocess
from .base import Backend


class LocalBackend(Backend):
    def submit(
        self,
        *,
        name,
        command,
        time,
        output_log,
        error_log,
        depends_on=None,
    ):
        print(f"▶ [LOCAL] {name}")
        print("  ", " ".join(command))
        subprocess.run(command, check=True)
        return None
