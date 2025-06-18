# duots

**duots** is a lightweight Python package for calculating features from paired time series signals — like those collected from symmetrical body parts (e.g., left and right wrists). It provides a composable, lazy, and efficient pipeline to build complex signal analysis routines without relying on heavy external libraries like NumPy or pandas.

---

## Features
-  **Composable feature pipelines** using functional programming
-  Modular primitives: segmentation, transformation, timeseries ops, value aggregation
-  Efficient via `functools.lru_cache` (minimizes redundant computation)
-  **Minimal dependencies**: uses only `scipy`
-  Designed for **paired signals** (e.g., `(left, right)` or `(x, y)`)
-  Easy to extend, debug, and test

## Design Philosophy
- **Composable**: Build powerful feature extractors from simple, small functions.
- **Efficient**: Shared operations are cached; performance scales with reuse.
- **Minimal**: Only `scipy` is used for essential math; avoids heavy dependencies.

## 📦 Installation
### From PyPI
```bash
pip install duots
```
### From Source
```bash
git clone https://github.com/4d30/duots.git
cd duots
pip install .
```

