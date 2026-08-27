"""Multi-start results (spec 15)."""

from dataclasses import dataclass, replace

import jax.numpy as jnp

from ..postprocess import align_to
from .result import Reconstruction


@dataclass(frozen=True)
class Ensemble:
    """A sequence of reconstructions that behaves like a Python collection.

    Runs are *not* mutually aligned as they come out of the solver: each one is
    the object up to its own global phase, translation and twin (spec 14.1).
    Compare them by error, or call :meth:`align` before doing anything that
    combines them.

    ``problem`` is carried so that a combined result can be scored against the
    original measurement; it is absent for hand-built ensembles.
    """

    runs: tuple
    problem: object = None

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Ensemble(self.runs[index], self.problem)
        return self.runs[index]

    @property
    def ranking(self) -> tuple:
        """Run indices ordered best (lowest error) first."""
        return tuple(sorted(range(len(self.runs)), key=lambda i: self.runs[i].error))

    @property
    def best(self) -> Reconstruction:
        return self.runs[self.ranking[0]]

    def align(self) -> "Ensemble":
        """Bring every run onto the best one, in phase, position and branch.

        The best run supplies the convention, nothing more: if it happens to sit
        in the conjugate-flipped branch, the whole aligned ensemble does too.
        That is fine for combining runs, which is what this is for, and it is not
        a statement about which branch is real (spec 14.1).
        """
        reference = self.best.object
        aligned = tuple(
            replace(run, object=align_to(run.object, reference)) for run in self.runs
        )
        return Ensemble(aligned, self.problem)

    def average(self) -> Reconstruction:
        """Coherent mean of the aligned runs.

        Aligning first is not optional: averaging raw runs cancels them against
        each other, because they differ by exactly the symmetries the solver
        cannot see.
        """
        aligned = self.align()
        mean = sum(run.object for run in aligned) / len(aligned)
        return Reconstruction(mean, self.best.support, self._score(mean))

    def _score(self, obj) -> jnp.ndarray:
        if self.problem is None:
            return jnp.zeros((0,), jnp.float32)
        from ..operators.modulus import modulus_project

        return jnp.atleast_1d(modulus_project(obj, self.problem)[1])
