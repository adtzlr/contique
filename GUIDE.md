# Theory Guide

A solution curve for `(n)` equilibrium equations `fun` in terms of `(n)` unknowns `x` and a load-proportionality-factor `lpf` should be found by numeric continuation from an initial equilibrium state `fun(x0, lpf0) = 0`. One additional equilibrium equation is added to the equations to ensure `(n+1)` equations in terms of `(n+1)` unknowns. The numeric solution is divided into three categories:

- Step
 + Cycle
  * Newton-Rhapson - Iteration

## Control Equation
For every 