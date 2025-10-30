<p align="center">
  <img height="160" alt="Image" src="https://github.com/user-attachments/assets/91912121-35df-4fac-a950-2fe4f75f51a9"/>
  <p align="center">Numerical continuation of nonlinear equilibrium equations.</p>
</p>

[![PyPI version shields.io](https://img.shields.io/pypi/v/contique.svg)](https://pypi.python.org/pypi/contique/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/contique.svg)](https://pypi.python.org/pypi/contique/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7931300.svg)](https://doi.org/10.5281/zenodo.7931300)
[![codecov](https://codecov.io/gh/adtzlr/contique/branch/main/graph/badge.svg?token=CXKRL8TLQY)](https://codecov.io/gh/adtzlr/contique)

Contique is a Python 3.9+ package that provides methods for numeric continuation.

## ✨ Highlights
Contique's numeric continuation method is best classified as a 

- ✅ **component-based continuation** with an adaptive 
- ✅ **magnitude-based control-component switching**.

> [!NOTE]
> *What is numeric continuation?* A solution curve for `(n)` equilibrium equations `fun` in terms of `(n)` unknowns `x` and a load-proportionality-factor `lpf` should be found by numeric continuation from an initial equilibrium state `fun(x0, lpf0) = 0`. 

## 📦 Installation
Install Python, open a terminal and run

```shell
pip install contique
```

Contique depends on
- `numpy` (for arrays) and
- `scipy` (check if matrix is sparse and for a sparse solver).

## 🚀 Getting Started
A given set of equilibrium equations in terms of `x` and `lpf` (a.k.a. load-proportionality-factor) should be solved by numeric continuation of a given initial solution. We start with the definition of a function

```python
import numpy as np


def fun(x, lpf, a, b):
    return np.array(
        [-a * np.sin(x[0]) + x[1] ** 2 + lpf, -b * np.cos(x[1]) * x[1] + lpf]
    )
```

with its initial solution

```python
x0 = np.zeros(2)
lpf0 = 0.0
```

and function parameters.

```python
a = 1
b = 1
```

Now let's run `contique.solve()` and plot the states of equilibrium.

```python
import contique

res = contique.solve(
    fun=fun,
    x0=x0,
    args=(a, b),
    lpf0=lpf0,
    dxmax=0.1,
    dlpfmax=0.1,
    maxsteps=75,
    maxcycles=4,
    maxiter=20,
    tol=1e-8,
    overshoot=1.05,
)
```

For each `step` a summary is printed out per `cycle`. This contains an information about the control component at the beginning and the end of a cycle as well as the norm of the residuals along with needed Newton-Rhapson `iterations` per `cycle`. As an example the ouput of some interesting `steps` 31-33 and 38-40 are shown below. The last column contains messages about the solution. On the one hand, in `step` 32, `cycle` 1 the control component changed from `+1` to `-2`, but the relative overshoot on the final control component `-2` was inside the tolerated range of `overshoot=1.05`. Therefore the solver proceeds with `step` 33 without re-cycling `step` 32. On the other hand, in `step` 39, `cycle` 1 the control component changed from `-2` to `-1` and this time the overshoot on the final control component `-1` was outside the tolerated range. A new `cycle` 2 is performed for `step` 39 with the new control component `-1`.

```markdown
|Step,C.| Control Comp. | Norm (Iter.#) | Message     |
|-------|---------------|---------------|-------------|

(...)

|  31,1 |   +1  =>   +1 | 7.6e-10 ( 3#) |             |
|  32,1 |   +1  =>   -2 | 1.7e-14 ( 4#) |tol.Overshoot|
|  33,1 |   -2  =>   -2 | 4.8e-12 ( 3#) |             |

 (...)
 
|  38,1 |   -2  =>   -2 | 9.2e-12 ( 3#) |             |
|  39,1 |   -2  =>   -1 | 1.9e-13 ( 3#) | => re-Cycle |
|     2 |   -1  =>   -1 | 2.3e-13 ( 4#) |             |
|  40,1 |   -1  =>   -1 | 7.9e-09 ( 3#) |             |

(...)
```

Next, we have to assemble the results

```python
X = np.array([r.x for r in res])
```

and plot the solution curve.

```python
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, 1], "C0.-")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.plot([0], [0], "C0o", lw=3)
plt.arrow(
    X[-2, 0],
    X[-2, 1],
    X[-1, 0] - X[-2, 0],
    X[-1, 1] - X[-2, 1],
    head_width=0.075,
    head_length=0.15,
    fc="C0",
    ec="C0",
)
plt.gca().set_aspect("equal")
```

[![Equilibrium-Equations-SinCos](https://github.com/adtzlr/contique/assets/5793153/20fb6415-d226-4859-b818-4f79194ba1e2)](https://github.com/adtzlr/contique/blob/main/tests/test_sincos.py)

Fig. 2 Solution states of [equilibrium equations](https://github.com/adtzlr/contique/blob/main/tests/test_sincos.py) solved with [contique](https://github.com/adtzlr/contique/blob/main/tests/test_sincos.py).

## 📝 A Brief Theory Guide - Numeric Continuation
[![Archimedean-Spiral](https://github.com/adtzlr/contique/assets/5793153/6b38c783-bdfc-470a-8a66-82a3ca663407)](https://github.com/adtzlr/contique/blob/main/tests/test_archimedean_spiral.py)

Fig. 1 [Archimedean spiral](https://en.wikipedia.org/wiki/Archimedean_spiral) equation solved with [contique](https://github.com/adtzlr/contique/blob/main/tests/test_archimedean_spiral.py).
  
### Extended Equilibrium Equations
The `lpf` value is appended to the unknows `x` which gives the so-called extended unknowns `y = [x, lpf]`. One additional control equation is added to the equilibrium equations to ensure `(n+1)` equations in terms of `(n+1)` extended unknowns (see next section). This reduces the solution to a point on the initial solution curve.

### Control Equation
The control equation is defined as follows: First, a needle-vector with dimension `(n+1)` is created and filled with zeros `needle = 0`. For a given initial signed control component `j` the needle is positioned at `needle[|j|] = 1`. The maximum allowed values per component are calculated as `ymax = y0 + np.sign(j) dymax`. The control equation is finally formulated as `f(y) = needle.T (y - ymax)`.

### Solution Technique
The numeric solution process is divided into three main parts:

- **Step**
    + Cycle
        * *Iteration* (...of a Newton-Rhapson root method)
  
As the name implies, a **Step** tries to find the extended unknowns for the next step forward of the equilibrium state. For each Cycle, the initial control component has to be evaluated first (see comment below). The additional control equation is evaluated with this initial control component. The generated extended equilibrium equations in terms of the extended unknows are now solved with the help of a root method (Newton-Rhapson *Iterations*). The solution of the root method `dy` is further normalized as `dy/dymax` and the final control component is evaluated as `j = |j| sign((dy/dymax)[|j|])` with `|j| = argmax(|dy/dymax|)`. If the control component changed, another Cycle is performed with the initial control component being now the final control component of the last cycle. This Cycle-loop is repeated until the control component does not change anymore.

> **Note**
> Pre-evaluation of the initial control component of a **Step**: This is performed by the linear solution of the extended equilibrium equations. It is equal to the result of the first *Iteration* of the Newton-Rhapson root method.

## 📄 Changelog
All notable changes to this project will be documented in [this file](CHANGELOG.md). The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 🔓 License
Contique - Numerical continuation of nonlinear equilibrium equations (C) 2021-2025 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.