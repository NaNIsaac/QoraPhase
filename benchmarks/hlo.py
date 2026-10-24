"""What XLA compiles around the FFTs of the model, read from the optimized program.

    uv run python benchmarks/hlo.py            # Markdown on stdout
    uv run python benchmarks/hlo.py --dump DIR # also write each compiled program

For ``FFT.adjoint(FFT.forward(x))`` and the bare ``ifftn(fftn(x))`` it lists the top-level
instructions of the compiled entry computation that are not FFTs, and how many arrays of
the input's size they write between them.  Each top-level instruction writes its result to
memory, so this counts the passes over the data that are not part of the transforms.  No
timing: the result does not depend on machine load.
"""

import argparse
import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp

from qoraphase.models.fft import FFT

SHAPES = [(256, 256), (1024, 1024), (64, 64, 64), (128, 128, 128), (65, 64, 63)]
INSTRUCTION = re.compile(r"^\s*(?:ROOT )?%\S+ = (\w+)\[([\d,]*)\]\S* ([\w-]+)\(", re.MULTILINE)
FREE = {"parameter", "constant", "fft", "bitcast", "tuple", "get-tuple-element"}


def entry(text):
    start = text.index("\nENTRY")
    return text[start : text.index("\n}", start)]


def passes(text, size):
    ops, written = [], 0
    for _dtype, dims, op in INSTRUCTION.findall(entry(text)):
        if op in FREE:
            continue
        ops.append(op)
        written += math.prod(int(d) for d in dims.split(",") if d)
    return ops, written / size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, help="directory for the compiled programs")
    args = parser.parse_args()
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)

    fft = FFT()
    cases = {
        "forward+adjoint": lambda a: fft.adjoint(fft.forward(a)),
        "raw fftn+ifftn": lambda a: jnp.fft.ifftn(jnp.fft.fftn(a, norm="ortho"), norm="ortho"),
    }
    print(f"JAX {jax.__version__}, {jax.devices()[0]}, complex64\n")
    print("| shape | case | FFTs | other instructions | arrays written outside the FFTs |")
    print("|---|---|---:|---:|---:|")
    for shape in SHAPES:
        x = jnp.zeros(shape, jnp.complex64)
        for name, fn in cases.items():
            text = jax.jit(fn).lower(x).compile().as_text()
            if args.dump:
                tag = f"{name.split()[0].replace('+', '_')}_{'x'.join(map(str, shape))}"
                (args.dump / f"{tag}.hlo.txt").write_text(text)
            ops, arrays = passes(text, math.prod(shape))
            ffts = len(re.findall(r" fft\(", entry(text)))
            print(f"| {'x'.join(map(str, shape))} | {name} | {ffts} | {len(ops)} | {arrays:g} |")


if __name__ == "__main__":
    main()
