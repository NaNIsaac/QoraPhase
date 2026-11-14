# Array conventions

| | |
|---|---|
| **Origin** | Every public array is **center-origin**: index `n // 2` is q=0. `ifftshift` is internal; never call it yourself. |
| **`intensity`** | A measured **intensity**, not an amplitude. Internally `sqrt(clip(intensity, 0))`. |
| **`mask`** | `True` marks an **unusable** pixel (beamstop, hot, saturated). Masked pixels are left unconstrained — never forced to zero. NaN or Inf is refused even under the mask: set those pixels to 0. On `main`, `diagnose` excludes masked pixels from peak and negative-value checks and warns when masked values are non-finite; the 0.2.0 release on PyPI still counts masked values in its diagnostics. |
| **`support`** | `True` is inside the object. Omit it and an autocorrelation support is derived (loose; pass a real one when you have it). |

Coming from a detector stack (for example [CXD](https://github.com/NaNIsaac/CXD)'s
`to_qoraphase`): [input contract](../input-contract.md) lists what the arrays must
satisfy, which coordinates stay with the caller, and the effects of cropping, binning and
interpolation.
