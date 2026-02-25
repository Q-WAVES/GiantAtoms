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
zeta=1.5

delta12 = -0.71
delta21 = -0.71
delta11 = 0.71
delta22 = -1.92
N = 100
dx=2
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


# Plot the dynamics of the fg and ee levels in different colors
colors = ['C0', 'C1']
plt.plot(ts, Ns2[0], label=r'$\Delta_1=%.2f, \Delta_2=%.2f, | fg \rangle $' % (delta21, delta22), color=colors[0], linewidth=3)
plt.plot(ts, EE, label=r'$\Delta_1=%.2f, \Delta_2=%.2f, | ee \rangle $' % (delta11, delta12), color=colors[1], linewidth=3)

plt.title(r"Atom dynamics, $g/J = %.3f, dx=%i$" % (g, dx))
plt.xlabel(r'$tJ$')
plt.ylabel(r'$|C_e(t)|^2$')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.show()
