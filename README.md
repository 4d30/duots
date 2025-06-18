# duots

**duots** is a lightweight Python package for calculating features from paired time series signals — like those collected from symmetrical body parts (e.g., left and right wrists). It provides a composable, lazy, and efficient pipeline to build complex signal analysis routines.

---

## Features
-  **Composable feature pipelines** using functional programming
-  Modular primitives: segmentation, transformation, timeseries ops, value aggregation
-  Efficient via `functools.lru_cache` (minimizes redundant computation)
-  **Minimal dependencies**: uses only `scipy` (and optionally [sampen-gpu](https://github.com/4d30/sampen)) 
-  Designed for **paired signals** (e.g., `(left, right)` or `(x, y)`)
-  Easy to extend, debug, and test

## Design Philosophy
- **Composable**: Build powerful feature extractors from simple, small functions.
- **Efficient**: Shared operations are cached; performance scales with reuse.
- **Minimal**: Only `scipy` is used for FFT, skew, and kurtosis; avoids heavy dependencies.

## 📦 Installation
### From PyPI  --  [duots](https://pypi.org/project/duots/)
```bash
pip install duots
```
### From Source
```bash
git clone https://github.com/4d30/duots.git
cd duots
pip install .
```

## Example Usage
```python
from duots import generate, compose

# Create a pair of signals, e.g., (left, right)
# They must be tuples, of the same length, without NaNs
sig_a = tuple(range(1, 100))
sig_b = tuple(range(1, 100))
signal_pair = (sig_a, sig_b,)

# Calculate values for each process
for proc in generate.processes():
    names, funcs = zip(*proc)
    name = compose.descriptors(names)
    composed_function = compose.functions(funcs)
    value = composed_function(signal_pair)
    print(f"{name}: {value}")
```
    



