"""Multi-start results."""

from dataclasses import dataclass, replace
from functools import cached_property

import jax.numpy as jnp

from ..postprocess import align_to
from .result import Reconstruction


@dataclass(frozen=True)
class Ensemble:
    """A sequence of reconstructions that behaves like a Python collection.

    Runs are *not* mutually aligned as they come out of the solver: each one is
    the object up to its own global phase, translation and twin.
    Compare them by error or by agreement (:attr:`consensus`), or call
    :meth:`align` before doing anything that combines them.

    ``problem`` is carried so that a combined result can be scored against the
    original measurement; it is absent for hand-built ensembles.  ``history``
    holds the earlier generations of a :func:`~qoraphase.evolve` run, oldest first.
    """

    runs: tuple
    problem: object = None
    history: tuple = ()

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
        """The run with the lowest Fourier error.

        Runs stuck in different basins can fit the data equally well: on BCDI
        benchmark data the phase error of single runs split into two groups
        (about 0.08 and 0.2 rad) whose Fourier errors differed by 1e-3, so the
        lowest error picked either.  :attr:`consensus` does not have that problem.
        """
        return self.runs[self.ranking[0]]

    @cached_property
    def spread(self) -> tuple:
        """Mean distance of each run from the others, after aligning them onto it.

        ``||align_to(other, run) - run|| / ||run||``, averaged over the other runs.
        A run in the basin most starts reach is close to the rest.
        """
        objects = [run.object for run in self.runs]
        spread = []
        for i, reference in enumerate(objects):
            norm = jnp.linalg.norm(reference)
            distances = [
                float(jnp.linalg.norm(align_to(other, reference) - reference) / norm)
                for j, other in enumerate(objects)
                if j != i
            ]
            spread.append(sum(distances) / len(distances) if distances else 0.0)
        return tuple(spread)

    @property
    def consensus(self) -> Reconstruction:
        """The run closest to all the others (smallest :attr:`spread`).

        Needs no ground truth, only agreement between starts.  With fewer than
        three runs there is no majority, and this is :attr:`best`.
        """
        if len(self.runs) < 3:
            return self.best
        return self.runs[min(range(len(self.runs)), key=self.spread.__getitem__)]

    def align(self, upsample: int = 1, reference: Reconstruction | None = None) -> "Ensemble":
        """Bring every run onto *reference* (default :attr:`best`), in phase, position and branch.

        The reference supplies the convention, nothing more: if it happens to sit
        in the conjugate-flipped branch, the whole aligned ensemble does too.
        That is fine for combining runs, which is what this is for, and it is not
        a statement about which branch is real.

        ``upsample`` is passed to :func:`~qoraphase.postprocess.align_to`; the
        default aligns to whole pixels.
        """
        reference = (self.best if reference is None else reference).object
        aligned = tuple(
            replace(run, object=align_to(run.object, reference, upsample)) for run in self.runs
        )
        return Ensemble(aligned, self.problem)

    def average(self, upsample: int = 1) -> Reconstruction:
        """Coherent mean of the runs aligned onto :attr:`consensus`.

        Aligning first is not optional: averaging raw runs cancels them against
        each other, because they differ by exactly the symmetries the solver
        cannot see.  The reference is the consensus run rather than the best one:
        a lowest-error run from a minority basin would set the branch and
        position, and the majority would then be averaged onto it.
        """
        consensus = self.consensus
        aligned = self.align(upsample, consensus)
        mean = sum(run.object for run in aligned) / len(aligned)
        return Reconstruction(mean, consensus.support, self._score(mean))

    def _score(self, obj) -> jnp.ndarray:
        if self.problem is None:
            return jnp.zeros((0,), jnp.float32)
        from ..operators.modulus import modulus_project

        return jnp.atleast_1d(modulus_project(obj, self.problem)[1])
