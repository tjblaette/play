import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def plot_sensitivities():
    """
    Create a line plot of the sensitivity achieved
    over increasing training epochs. Input are
    sensitivity.tsv files passed as command line
    arguments, which contain the run ID (network_dir) 
    in the header, the epoch in the index column and the
    achieved sensitivity [0, 1.0] in the only data column.
    """
    dfs = []
    for filename in sys.argv[1:]:
        print(filename)
        df = pd.read_csv(filename, sep='\t', header=0, index_col=0)
        dfs.append(df)

    zz = pd.concat(dfs, axis=1)
    zz['Epoch'] = zz.index

    zz_long = pd.melt(
        zz,
        id_vars=['Epoch'],
        var_name='Run-ID', 
        value_name='Sensitivity')

    # remove the timestamp to group replicates with an
    # otherwise identical run ID / experiment setting
    zz_long['Run-ID'] = ['_'.join(run_id.split('_')[:-1]) 
        for run_id in zz_long['Run-ID']]

    sns.lineplot(
        x='Epoch', 
        y='Sensitivity', 
        data=zz_long, 
        hue='Run-ID', 
        dashes=False) #with dashes, max 6 colors available!
    plt.show()

plot_sensitivities()
