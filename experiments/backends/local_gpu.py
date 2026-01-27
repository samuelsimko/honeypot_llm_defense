import os
import uuid
import subprocess
import threading
import queue
import time
import random
from typing import List, Optional, Dict

from .base import Backend



# --------------------------------------------------
# Helpers
# --------------------------------------------------

def detect_visible_gpus() -> List[str]:
    """
    Return a list of GPU IDs visible to this process.
    Respects CUDA_VISIBLE_DEVICES if set.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if visible == "":
            return []
        return [x.strip() for x in visible.split(",") if x.strip() != ""]
    out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
    # one GPU per line
    n = out.count("\n")
    return [str(i) for i in range(n)]


def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------
# Internal Job Object
# --------------------------------------------------

class _Job:
    def __init__(
        self,
        job_id: str,
        name: str,
        command: List[str],
        output_log: str,
        error_log: str,
        depends_on: Optional[List[str]],
    ):
        self.job_id = job_id
        self.name = name
        self.command = command
        self.output_log = output_log
        self.error_log = error_log
        self.depends_on = list(depends_on or [])

        self.completed = threading.Event()
        self.failed = False
        self.returncode: Optional[int] = None
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None


# --------------------------------------------------
# Backend
# --------------------------------------------------

class LocalGPUBackend(Backend):
    """
    Single-node GPU worker backend.
    Each GPU is treated as a worker that runs jobs sequentially.

    - Uses a single shared queue.
    - Workers requeue blocked jobs (waiting on deps).
    - Writes a global scheduler log to runs/scheduler.log.
    """

    def __init__(self, num_gpus: Optional[int] = None):
        self.visible_gpus = detect_visible_gpus()

        if num_gpus is not None:
            self.visible_gpus = self.visible_gpus[:num_gpus]

        if not self.visible_gpus:
            raise RuntimeError("No GPUs detected (CUDA_VISIBLE_DEVICES empty and nvidia-smi -L returned none).")

        self.jobs: Dict[str, _Job] = {}
        self.job_queue: queue.Queue[_Job] = queue.Queue()

        os.makedirs("runs", exist_ok=True)
        self.scheduler_log_path = "runs/scheduler.log"
        self._log_fp = open(self.scheduler_log_path, "a", buffering=1)

        self._log(f"[INIT] LocalGPUBackend with GPUs: {self.visible_gpus}")
        self._start_workers()

    def wait_all(self):
        self._log("[WAIT] waiting for all jobs to finish")
        self.job_queue.join()
        self._log("[DONE] all jobs finished")

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._log_fp.write(line + "\n")

    # --------------------------------------------------
    # Public API (matches SlurmBackend signature)
    # --------------------------------------------------

    def submit(
        self,
        *,
        name: str,
        command: List[str],
        time: str,  # kept for interface compatibility (ignored here)
        output_log: str,
        error_log: str,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        job_id = str(uuid.uuid4())[:8]

        job = _Job(
            job_id=job_id,
            name=name,
            command=command,
            output_log=output_log,
            error_log=error_log,
            depends_on=depends_on,
        )

        self.jobs[job_id] = job
        self.job_queue.put(job)

        self._log(f"[QUEUE] job={job_id} name={name} deps={depends_on or []}")
        return job_id

    # --------------------------------------------------
    # Internal: workers
    # --------------------------------------------------

    def _start_workers(self):
        for worker_id, gpu_id in enumerate(self.visible_gpus):
            t = threading.Thread(
                target=self._gpu_worker,
                args=(worker_id, gpu_id),
                daemon=False,
            )
            t.start()
            self._log(f"[WORKER-{worker_id}] started on GPU {gpu_id}")

    def _gpu_worker(self, worker_id: int, gpu_id: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["NVIDIA_VISIBLE_DEVICES"] = gpu_id
        env.update({
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
        })

        while True:
            job: _Job = self.job_queue.get()

            # -------------------------
            # Dependency checks (non-blocking)
            # -------------------------
            blocked_on = None
            failed_dep = None

            for dep in job.depends_on:
                parent = self.jobs.get(dep)
                if parent is None:
                    # unknown dep id: treat as satisfied
                    continue
                if parent.failed:
                    failed_dep = dep
                    break
                if not parent.completed.is_set():
                    blocked_on = dep
                    break

            if failed_dep is not None:
                job.failed = True
                job.returncode = None
                job.completed.set()
                self._log(f"[SKIP] job={job.job_id} name={job.name} failed_dep={failed_dep}")
                self.job_queue.task_done()
                continue

            if blocked_on is not None:
                # requeue
                self._log(f"[REQUEUE] job={job.job_id} name={job.name} waiting_on={blocked_on}")
                self.job_queue.put(job)
                self.job_queue.task_done()
                time.sleep(0.15 + random.random() * 0.15)
                continue

            # -------------------------
            # Run job
            # -------------------------
            self._run_job(worker_id, gpu_id, job, env)
            self.job_queue.task_done()

    def _run_job(self, worker_id: int, gpu_id: str, job: _Job, env: Dict[str, str]):
        job.started_at = time.time()
        self._log(f"[START] job={job.job_id} name={job.name} gpu={gpu_id} out={job.output_log} err={job.error_log}")

        _ensure_dir_for(job.output_log)
        _ensure_dir_for(job.error_log)

        wrapped_cmd = (
            "set -euo pipefail; "
            "source /root/honeypot_llm_defense/venv/bin/activate; "
            "source .env; "
            "nvidia-smi; "
            "sleep 1; "
            + " ".join(job.command)
        )

        with open(job.output_log, "w") as out, open(job.error_log, "w") as err:
            ret = subprocess.run(
                ["bash", "-lc", wrapped_cmd],
                env=env,
                stdout=out,
                stderr=err,
            )

        job.ended_at = time.time()
        job.returncode = ret.returncode
        dt = job.ended_at - (job.started_at or job.ended_at)

        if ret.returncode != 0:
            job.failed = True
            self._log(f"[END] job={job.job_id} name={job.name} gpu={gpu_id} rc={ret.returncode} dt={dt:.1f}s STATUS=FAIL")
        else:
            self._log(f"[END] job={job.job_id} name={job.name} gpu={gpu_id} rc=0 dt={dt:.1f}s STATUS=OK")

        job.completed.set()
