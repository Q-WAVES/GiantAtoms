import sys
import os

sys.path.append(os.path.abspath("../../src"))

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm
from GA import *
import seaborn as sns
import matplotlib as mpl


# Latex font
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = [7, 6]
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'times',
    'text.latex.preamble': r'''
        \usepackage{times}
        \usepackage{amsmath,amssymb,bm}
        \DeclareSymbolFont{letters}{OML}{cmm}{m}{it}
    '''
})

# Settings
nAtoms = 2
J = 1
g = 0.1
zeta=1

delta12 = -1
delta21 = -0.98
delta11 = 1
delta22 = -2.5
N = 100
dx = 2
CouplePoints = [[48,50,52], [49,51,53]]
nT, maxT = 1000, 100

# Construct two photon GA object TGA and compute the eigenvalues/vectors of the Hamiltonian
giantAtom = TGA(N, J, nAtoms, [[g, zeta*g, g], [g, zeta*g, g]], [delta12, delta22], [delta11, delta21], CouplePoints)
es, vs = giantAtom.computeEigens()
ts = np.linspace(0, maxT, nT)

# Initial state with population in atom 1 and nothing in the cavities
initialState = giantAtom.constructInitialState(atoms=[1, 2], cavities=[0])

# Calculate the dynamics for level f, e, and ee
Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)
print(np.min(EE))

# Plot the dynamics of the fg and ee levels in different colors
import matplotlib.pyplot as plt

# Global style for large fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 28,
    "axes.titlesize": 32,
    "axes.labelsize": 30,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 24,
})

colors = ['C0', 'C1', 'C2', 'C3']

plt.figure(figsize=(6,5))

plt.plot(ts, Ns2[0], label=r'$n^{(20)}$', color=colors[0], linewidth=4)
plt.plot(ts, EE, label=r'$n^{(11)}$', color=colors[1], linewidth=4)

plt.title(r'$\omega_{DF}/J=\pm 1,\ \Delta x=2$')
plt.xlabel(r'$tJ$')
plt.ylabel(r'$n(t)$')

plt.ylim(-0.1, 1.1)

plt.xticks([0, 50, 100])
plt.yticks([0, 0.5, 1])

plt.legend(loc='upper right',framealpha=0.6)
plt.tight_layout()

plt.savefig("Fig3c.svg", dpi=300)
plt.show()
