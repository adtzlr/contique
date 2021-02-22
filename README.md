# contique
Numeric **conti**nuation of e**qu**ilibrium **e**quations

[![PyPI version shields.io](https://img.shields.io/pypi/v/contique.svg)](https://pypi.python.org/pypi/contique/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/contique.svg)](https://pypi.python.org/pypi/contique/)
![Made with love in Graz](https://madewithlove.now.sh/at?heart=true&colorA=%233b3b3b&colorB=%231f744f&text=Graz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Example
A given set of equilibrium equations in terms of `x` and `lpf` (a.k.a. load-proportionality-factor) should be solved by numeric continuation of a given initial solution.


### Function definition
```python
def fun(x, lpf, a, b):
    return np.array([-a * np.sin(x[0]) + x[1]**2 + lpf, 
	                 -b * np.cos(x[1]) * x[1]      + lpf])
```

with it's initial solution
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
	dxmax=0.05,
	dlpfmax=0.05,
	maxsteps=80,
	maxcycles=4,
	maxiter=20,
	tol=1e-6,
)
```

Assemble results

```python
X = np.array([res.x for res in Res])
```

and plot the solution curve.

```python
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, 1], ".-")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
```

<img src="https://raw.githubusercontent.com/adtzlr/contique/main/test/test_sincos.svg" width="30%">