import numpy as np

nvals = np.logspace(3, 13, base=2, num=10, dtype=int)

def create_dirac_delta_spikes(xvals):
    p = np.zeros(xvals.shape)
    spike_positions = [-0.25, 0.25]
    for spike_pos in spike_positions:
        closest_index = np.argmin(np.abs(xvals - spike_pos))
        p[closest_index] = 1
    p = p / np.sum(p)
    return p

def create_parabolic_distribution(xvals):
    p = 4 * (xvals - 1) * (xvals - 1)
    p /= np.sum(p)
    return p

distribution_descriptions = [
    'Gaussian',
    'Bimodal Gaussian',
    'Cosine',
    'Sine',
    'Exponential',
    'Logarithmic',
    'Dirac Delta Spikes',
    'Uniform',
    'Step Function',
    'Parabolic Dist.'
]

distributions = [
    lambda xvals: (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * xvals**2),
    lambda xvals: 0.5*np.exp(-(xvals+0.5)**2) + 0.5*np.exp(-(xvals-0.5)**2),
    lambda xvals: np.cos(np.pi*xvals) + 1,
    lambda xvals: np.sin(np.pi*xvals)+1,
    lambda xvals: np.exp(-np.abs(xvals)),
    lambda xvals: -np.log(np.abs(xvals + 0.1)),
    create_dirac_delta_spikes,
    lambda xvals: np.ones(xvals.shape),
    lambda xvals: (xvals > 0).astype(float),
    create_parabolic_distribution
]