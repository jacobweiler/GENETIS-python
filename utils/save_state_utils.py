# savestate class
import yaml
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Union, Optional

log = logging.getLogger(__name__)


@dataclass
class SaveState:
    generation: int = 0
    step: str = "ga"
    timestamp: str = datetime.now().isoformat()

    @classmethod
    def __str__(self):
        return (
            f"SaveState(generation={self.generation}, step='{self.step}', "
            f"timestamp='{self.timestamp}')"
        )

    def load(cls, filepath: Union[str, Path]):
        filepath = Path(filepath)
        if filepath.exists():
            log.info(f"Loading savestate from {filepath}")
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            return cls(**data)
        else:
            log.info(f"No savestate found at {filepath}, creating new one")
            state = cls()
            state.save(filepath)
            return state

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            yaml.dump(asdict(self), f, sort_keys=False)
        log.debug(f"Saved state to {filepath}")

    def update(
        self,
        step: str,
        generation: Optional[int] = None,
        filepath: Union[Path, str] = None,
    ):
        self.step = step
        if generation is not None:
            self.generation = generation
        if filepath:
            self.save(filepath)
            log.info(f"Updated state: gen={self.generation}, step={self.step}")
