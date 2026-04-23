from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RngKey:

    fold: int
    unit: int
    tau: int
    trial: Optional[int] = None


class RngManager:

    def __init__(self, master_seed: Optional[int] = None) -> None:
        self._ss = np.random.SeedSequence(master_seed)

    def _derive(self, *parts: int) -> np.random.Generator:
        if any(p < 0 for p in parts):
            raise ValueError('Seed parts must be non-negative integers.')
        child_ss = np.random.SeedSequence(
            entropy=self._ss.entropy,
            spawn_key=np.asarray(parts, dtype=np.uint32).tolist(),
        )
        return np.random.Generator(np.random.PCG64(child_ss))

    def rng_for(self, key: RngKey) -> np.random.Generator:
        if key.trial is None:
            return self._derive(key.fold, key.unit, key.tau)
        return self._derive(key.fold, key.unit, key.tau, key.trial)

    def for_transition(
        self,
        fold: int,
        unit: int,
        tau: int,
        trial: Optional[int] = None,
        particle: Optional[int] = None,
    ) -> np.random.Generator:
        parts = [fold, unit, tau]
        if trial is not None:
            parts.append(trial)
        if particle is not None:
            if trial is None:
                parts.append(0)
            parts.append(particle)
        return self._derive(*parts)

    def for_unit(self, fold: int, unit: int) -> np.random.Generator:
        return self.rng_for(RngKey(fold=fold, unit=unit, tau=0, trial=None))
