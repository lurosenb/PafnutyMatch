from utils import load_results, save_results, load_data

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as w1_dist

import cvxpy as cp

def create_grid(n, eps, C=10.0):
    step_size = 2 / (C * (n * eps))
    grid = np.arange(-1, 1 + step_size, step_size)
    return grid

def scale_and_bucket(data, grid, a, b):
    min_data = np.min(data)
    max_data = np.max(data)
    scaled_data = a + ((data - min_data) * (b - a) / (max_data - min_data))

    distribution = np.zeros(len(grid))

    for value in scaled_data:
        if value == 1:
            distribution[-1] += 1
        else:
            index = np.searchsorted(grid, value, side='left') - 1
            distribution[index] += 1

    return distribution

dataframes = load_data()

start = 3
end = 10
nvals = np.logspace(start, end, base=2, num=end-start, dtype=int)
small_increment = 1e-6
a, b = -1, 1
eps = 1.0
num_iterations = 10

results = []

for n in nvals:
  grid_points = create_grid(n, eps)
  m = len(grid_points)
  T = np.zeros((n+1, m))
  T_bar = np.zeros((n, m))
  T[0, :] = 1
  T[1, :] = grid_points

  for i in range(2, n+1):
    T[i, :] = 2 * grid_points * T[i-1, :] - T[i-2, :]

  for i in range(0, n):
    T_bar[i, :] = T[i+1, :] / ((np.pi / 2))

  for i, (col, dataframe) in enumerate(dataframes.items()):
    data = dataframe.astype(float)
    data = data.sample(n=n, random_state=i)
    p = scale_and_bucket(data, grid_points, a, b)
    for iteration in range(num_iterations):
        print(iteration)
        np.random.seed(iteration)

        all_mass = np.sum(p)
        p = p/all_mass
        scaled_mom = T_bar @ p

        delta = (1 / n**2)
        sigma_squared = (4 / np.pi) * (1 + np.log(n)) * np.log(1.25 / delta) / (n ** 2 * eps ** 2)

        scaled_mom += (np.random.randn(n) * np.sqrt(sigma_squared) * np.sqrt(1 + np.arange(n)))

        x = cp.Variable(m)
        objective = cp.Minimize(cp.sum(cp.square((T_bar @ x - scaled_mom)) / (np.arange(1, n+1) ** 2)))
        constraints = [x >= 0, cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        q = x.value

        # Due to numerical precision issues, need to
        # manually enforce non-negativity for values very close to 0
        mag_near_0 = np.sum(q[q < 0])

        q[q < 0] = 0
        wassdist = w1_dist(grid_points, grid_points, u_weights = p/np.sum(p), v_weights = q)

        results.append({
            'Column': col,
            'n': n,
            'Wasserstein Distance': wassdist,
            'Iteration': iteration
        })

    save_results(results, n, folder='caching_dp')

df = pd.DataFrame(results)
print(df)

