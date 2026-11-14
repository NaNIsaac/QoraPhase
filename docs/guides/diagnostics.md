# Check the measurement and reconstruction

Check the measurement against the assumptions the library makes about it:

```python
print(qp.diagnose(intensity, mask=mask, support=support))
```

It reports the centring offset two ways (brightest pixel, and Friedel symmetry,
which resolves half a pixel and works on patterns with no dominant peak), how
much of the array the autocorrelation fills, the negative and masked fractions,
and warns about each. An off-centre pattern is the most costly and least visible
failure.

`examples/diagnose.py` does the same from the command line, then reconstructs
and reports whether independent starts agree:

```
python examples/diagnose.py scan.npz --key intensity --support sup.npy --starts 8
```

A low Fourier error says a run fits the data. Agreement between starts after alignment checks repeatability, but does not by itself establish physical correctness or a unique solution.
