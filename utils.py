import pandas as pd
import numpy as np
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'no-latex', 'grid'])

def save_results(results, k):
  df = pd.DataFrame(results)
  filename = f'caching/results_up_to_k_{k}.pkl'
  with open(filename, 'wb') as f:
    pickle.dump(df, f)

def load_results(k):
  filename = f'caching/results_up_to_k_{k}.pkl'
  with open(filename, 'rb') as f:
    df = pickle.load(f)
  return df

def plot_from_results_df(nvals, df):
    colorblind_palette = ["#D81B60", "#1E88E5", "#FFC107", "#004D40"]
    sns.set_palette(sns.color_palette(colorblind_palette))
    
    plt.rcParams.update({'font.size': 14})

    distribution_descriptions = df['Distribution'].unique()

    for distribution in distribution_descriptions:
        plt.figure(figsize=(5, 3))

        sns.lineplot(data=df[df['Distribution'] == distribution], x='N', y='Wasserstein Distance', ci='sd', estimator='mean', linewidth=2.5, label='Experimental Result')

        nvals = df[df['Distribution'] == distribution]['N'].unique()

        plt.loglog(nvals, 1/nvals, label='1/n', linestyle='--', linewidth=1.5)
        plt.loglog(nvals, 1/np.sqrt(nvals), label='1/sqrt(n)', linestyle='--', linewidth=1.5)
        plt.loglog(nvals, np.log(nvals)/nvals, label='log(n)/n', linestyle='--', linewidth=1.5)

        plt.xlabel('k')
        plt.ylabel('Wasserstein Distance')
        plt.title(f'Distribution: {distribution}')
        plt.legend()

        filename = f'plots/{distribution.replace(" ", "_").lower()}.pdf'
        plt.savefig(filename, format='pdf')
        plt.show()
        print(f'Saved plot as {filename}')