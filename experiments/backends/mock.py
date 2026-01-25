# experiments/backends/mock.py

from typing import List, Optional

class MockBackend:
    def submit(
        self,
        *,
        name: str,
        command: List[str],
        time: Optional[str] = None,
        output_log: Optional[str] = None,
        error_log: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> Optional[str]:
        print("\n🧪 MOCK SUBMIT")
        print("Job name:", name)
        if depends_on:
            print("Depends on:", depends_on)
        if time:
            print("Time:", time)
        if output_log:
            print("Stdout:", output_log)
        if error_log:
            print("Stderr:", error_log)

        print("Command:")
        print(" ".join(command))
        print("─" * 80)

        # Return a fake job id so dependency logic still works
        return f"mock_{name}"
