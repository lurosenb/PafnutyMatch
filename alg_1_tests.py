## NOTE: To depracate, keeping for parity with paper submission.

from distributions import distributions, distribution_descriptions, nvals
from utils import load_results, save_results

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as w1_dist

import cvxpy as cp

a, b = [-1, 1]

results = []
num_iterations = 10

for k in nvals:
  gran = 1 / (2 * k)
  xvals = np.linspace(a + gran, b - gran, int(1/gran))
  m = len(xvals)
  T = np.zeros((k+1, m))
  T_bar = np.zeros((k, m))
  T[0, :] = 1
  T[1, :] = xvals

  for i in range(2, k+1):
    T[i, :] = 2 * xvals * T[i-1, :] - T[i-2, :]

  for i in range(0, k):
    T_bar[i, :] = T[i+1, :] / ((np.pi / 2))

  for distIdx, distFunc in enumerate(distributions):
    print(distribution_descriptions[distIdx])
    p = distFunc(xvals)
    p[p<0] = 0
    for iteration in range(num_iterations):
          print(iteration)
          n = np.sum(p)
          p = p/n
          scaled_mom = T_bar @ p
          
          scaled_mom += (np.random.randn(k)*np.sqrt(1 + np.arange(k))/k)

          x = cp.Variable(m)
          objective = cp.Minimize(cp.norm((T_bar @ x - scaled_mom) / (np.arange(1, k+1))))
          constraints = [x >= 0, cp.sum(x) == 1]
          prob = cp.Problem(objective, constraints)
          prob.solve(solver=cp.MOSEK, verbose=False)

          q = x.value

          # Due to numerical precision issues, need to
          # manually enforce non-negativity for values very close to 0
          mag_near_0 = np.sum(q[q < 0])

          q[q < 0] = 0
          wassdist = w1_dist(xvals, xvals, u_weights = p/np.sum(p), v_weights = q)

          results.append({
                'Distribution': distribution_descriptions[distIdx],
                'k': k,
                'Wasserstein Distance': wassdist,
                'Iteration': iteration
            })
          
  save_results(results, k)

df = pd.DataFrame(results)
print(df)