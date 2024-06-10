import matplotlib.pyplot as plt
import numpy as np
import pathlib

folder = pathlib.Path(__file__).parent / 'output'

files = folder.glob('*.txt')

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

    plt.plot(data, label=name, ls=ls, )

plt.legend()
plt.xlabel("Number of biases")
plt.ylabel(r"Performance $\left(\frac{\text{steps}}{\Delta t \cdot N}\right)$")
plt.yscale('log')
plt.savefig(folder / 'performance.png')