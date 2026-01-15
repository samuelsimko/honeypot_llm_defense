from abc import ABC, abstractmethod
from typing import List, Optional


class Backend(ABC):
    @abstractmethod
    def submit(
        self,
        *,
        name: str,
        command: List[str],
        time: str,
        output_log: str,
        error_log: str,
        depends_on: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Submit a job.
        Returns a job_id if applicable (SLURM), else None.
        """
        pass
