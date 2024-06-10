import matplotlib.pyplot as plt
import numpy as np
import pathlib
import matplotlib as mpl
import matplotlib.colors as mcolors
from hoomd.mimse import __version__ as mimse_version

folder = pathlib.Path(__file__).parent / 'output'

files = folder.glob('*.txt')

norm = mcolors.LogNorm(vmin=100, vmax=100_000)

for file in files:
    name = file.stem
    data = np.loadtxt(file)

    dev = name.split('_')[1]
    if dev == 'CPU':
        ls = '-'
    elif dev == 'GPU':
        ls = ':'
    else:
        ls = "--"

    N = int(name.split('_')[2].split('-')[1])

    c = plt.cm.viridis(norm(N))

    plt.plot(data, label=name, ls=ls, color=c)

plt.title(f"hoomd-mimse    tag={mimse_version}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Number of biases")
plt.ylabel(r"Performance $\left(\frac{\text{steps}}{\Delta t \cdot N}\right)$")
plt.yscale('log')
plt.savefig(folder / 'performance.png', bbox_inches='tight')