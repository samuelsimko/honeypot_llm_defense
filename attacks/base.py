from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class Attack(ABC):
    """
    Base class for all attacks.

    Lifecycle:
      1. __init__(...)
      2. initialize(...)          # optional global setup
      3. run_example(...)         # run attack for ONE prompt
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        **attack_config
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.attack_config = attack_config

        self.model.eval()
        self.model.to(self.device)

    def initialize(self, **kwargs) -> None:
            pass

    @abstractmethod
    def run_example(
        self,
        *,
        behavior_id: str,
        prompt: str,
        target: Optional[str] = None,
        variant_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run attack on a SINGLE example.

        Must return:
        {
          "prompt": str,
          "generated": str,
          "attack_metadata": dict (optional)
        }
        """
        pass