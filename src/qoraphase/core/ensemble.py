"""Multi-start results (spec 15)."""

from dataclasses import dataclass

from .result import Reconstruction


@dataclass(frozen=True)
class Ensemble:
    """A sequence of reconstructions that behaves like a Python collection.

    Runs are only comparable by their error metric: they are *not* aligned with
    each other, so averaging them without alignment is meaningless (spec 14.1).
    """

    runs: tuple

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Ensemble(self.runs[index])
        return self.runs[index]

    @property
    def ranking(self) -> tuple:
        """Run indices ordered best (lowest error) first."""
        return tuple(sorted(range(len(self.runs)), key=lambda i: self.runs[i].error))

    @property
    def best(self) -> Reconstruction:
        return self.runs[self.ranking[0]]
