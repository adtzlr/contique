# contique
Numeric **conti**nuation of nonlinear e**qu**ilibrium **e**quations

[![PyPI version shields.io](https://img.shields.io/pypi/v/contique.svg)](https://pypi.python.org/pypi/contique/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/contique.svg)](https://pypi.python.org/pypi/contique/)
![Code coverage](coverage.svg)
![Made with love in Graz](https://madewithlove.now.sh/at?heart=true&colorA=%233b3b3b&colorB=%231f744f&text=Graz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://raw.githubusercontent.com/adtzlr/contique/main/test/test_archimedean_spiral.svg" width="75%">

Fig. 1 [Archimedean spiral](https://en.wikipedia.org/wiki/Archimedean_spiral) equation solved with [contique](https://github.com/adtzlr/contique/blob/main/test/test_archimedean_spiral.py)

## Example
A given set of equilibrium equations in terms of `x` and `lpf` (a.k.a. load-proportionality-factor) should be solved by numeric continuation of a given initial solution.


### Function definition
```python
def fun(x, lpf, a, b):
    return np.array([-a * np.sin(x[0]) + x[1]**2 + lpf, 
                     -b * np.cos(x[1]) * x[1]    + lpf])
```

with its initial solution
```python
x0 = np.zeros(2)
lpf0 = 0.0
```

and function parameters
```python
a = 1
b = 1
```

### Run `contique.solve` and plot equilibrium states

```python
Res = contique.solve(
    fun=fun,
    x0=x0,
    args=(a, b),
    lpf0=lpf0,
    dxmax=0.1,
    dlpfmax=0.1,
    maxsteps=75,
    maxcycles=4,
    maxiter=20,
    tol=1e-6,
)
```

For each `step` a summary is printed out per cylce. This contains needed Newton-Rhapson `iterations` per `cycle` if the step was successful and an information about the control component at the beginning and the end of a cycle. As an example the ouput of Steps 71, 72 and 73 are shown below.

```markdown
| Step (Cycle) | Control Comp. | Equili. | Status        |
|--------------|---------------|---------|---------------|
|   71     (1) |   -1  =>   -1 | 1.1e-13 | Success ( 4#) |
|   72     (1) |   -1  =>   +2 | 1.4e-09 | Recycle       |
|          (2) |   +2  =>   +2 | 6.1e-11 | Success ( 3#) |
|   73     (1) |   +2  =>   +2 | 1.1e-10 | Success ( 3#) |
```

Next, we have to assemble the results

```python
X = np.array([res.x for res in Res])
```

and plot the solution curve.

```python
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, 1], "C0.-")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.plot([0],[0],'C0o',lw=3)
plt.arrow(X[-2,0],X[-2,1],X[-1,0]-X[-2,0],X[-1,1]-X[-2,1],
          head_width=0.075, head_length=0.15, fc='C0', ec='C0')
plt.gca().set_aspect('equal')
```

<img src="https://raw.githubusercontent.com/adtzlr/contique/main/test/test_sincos.svg" width="75%">

Fig. 2 Solution states of [equilibrium equations](https://github.com/adtzlr/contique/blob/main/test/test_sincos.py) solved with contique