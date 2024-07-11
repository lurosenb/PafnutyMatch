import pandas as pd
import numpy as np
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from folktables import ACSDataSource, ACSEmployment
from sklearn.datasets import fetch_california_housing
from ucimlrepo import fetch_ucirepo

import scienceplots

from distributions import create_dirac_delta_spikes, create_parabolic_distribution

plt.style.use(['science', 'no-latex', 'grid'])

distributions = {
    'Gaussian': lambda xvals: (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * xvals**2),
    'Bimodal Gaussian': lambda xvals: 0.5 * np.exp(-0.5 * (xvals + 0.5)**2) + 0.5 * np.exp(-0.5 * (xvals - 0.5)**2),
    'Cosine': lambda xvals: np.cos(np.pi * xvals) + 1,
    'Sine': lambda xvals: np.sin(np.pi * xvals) + 1,
    'Exponential': lambda xvals: np.exp(-np.abs(xvals)),
    'Dirac Delta Spikes': create_dirac_delta_spikes,
    'Uniform': lambda xvals: np.ones(xvals.shape),
    'Step Function': lambda xvals: (xvals > 0).astype(float),
    'Parabolic Dist.': create_parabolic_distribution,
    'Power Law': lambda xvals: np.power(xvals + 1.1, -2) 
}

def generate_samples(distribution_name, n_samples):
    xvals = np.linspace(-1, 1, 1000)
    if distribution_name not in distributions:
        raise ValueError(f"Unknown distribution: {distribution_name}")
    pdf_function = distributions[distribution_name]
    pdf_values = pdf_function(xvals)
    pdf_values /= np.sum(pdf_values)
    data = np.random.choice(xvals, size=n_samples, p=pdf_values)
    return pd.Series(data)

def load_data():
    dataframes = {}

    ## FAKE DATA FROM PDFS
    dists_to_gen_from = ['Gaussian', 'Sine', 'Power Law']
    n_samples = 5000
    
    for dist in dists_to_gen_from:
        dataframes[dist] = generate_samples(dist, n_samples)

    ## REAL DATA
    acs_data_col = 'PINCP'
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["NY"], download=True)
    acs_data.dropna(subset=[acs_data_col], inplace=True)
    dataframes[acs_data_col] = acs_data[acs_data_col]
    
    housing_data_col = 'HouseAge'
    housing_data = fetch_california_housing(as_frame=True).frame
    dataframes[housing_data_col] = housing_data[housing_data_col]

    diabetes_col = 'PhysHlth'
    diabetes = fetch_ucirepo(name='CDC Diabetes Health Indicators')
    X = diabetes.data.features
    dataframes[diabetes_col] = X[diabetes_col]

    return dataframes
  
def save_results(results, k, folder='caching'):
  df = pd.DataFrame(results)
  filename = f'{folder}/results_up_to_k_{k}.pkl'
  with open(filename, 'wb') as f:
    pickle.dump(df, f)

def load_results(k, folder='caching'):
  filename = f'{folder}/results_up_to_k_{k}.pkl'
  with open(filename, 'rb') as f:
    df = pickle.load(f)
  return df

def plot_from_results_df(nvals, df, x='N', data_name='Distribution'):
    colorblind_palette = ["#D81B60", "#1E88E5", "#FFC107", "#004D40"]
    sns.set_palette(sns.color_palette(colorblind_palette))
    
    plt.rcParams.update({'font.size': 16})

    distribution_descriptions = df[data_name].unique()

    for distribution in distribution_descriptions:
        plt.figure(figsize=(6, 4))

        sns.lineplot(data=df[df[data_name] == distribution], x=x, y='Wasserstein Distance', ci='sd', estimator='mean', linewidth=2.5, label='Alg. 2')

        nvals = df[df[data_name] == distribution][x].unique()
        
        C_1 = 4
        C_2 = 4
        C_3 = 2
        plt.loglog(nvals, C_1 * (1/nvals), label=r'$O(1/n)$', linestyle='--', linewidth=1.5)
        plt.loglog(nvals, C_2 * (1/np.sqrt(nvals)), label=r'$O(1/\sqrt{n})$', linestyle='--', linewidth=1.5)
        plt.loglog(nvals, C_3 * ((np.log(nvals) * np.sqrt(np.log(nvals)))/nvals), label=r'$O(log^{3/2}(n)/n)$', linestyle='--', linewidth=1.5)

        plt.xlabel(r'$n$')
        plt.ylabel(r'$W_1$ distance')
        plt.title(f'{distribution}')
        plt.legend()

        filename = f'plots/{distribution.replace(" ", "_").lower()}.pdf'
        plt.savefig(filename, format='pdf')
        plt.show()
        print(f'Saved plot as {filename}')