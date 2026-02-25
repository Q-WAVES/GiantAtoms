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

def DFIfreq(k, J):
    return -2 * J * np.cos(k)

J = 1
dx = 4
k = np.pi / dx    
deltadf1 = DFIfreq(k + (0 * np.pi) / dx, J)
deltadf2 = DFIfreq(k + (2 * np.pi) / dx, J)

deltas1 = [-deltadf1, -deltadf2]
deltas2 = [-deltadf2, -3 - deltadf2]


print(deltas1)
print(deltas2)   
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

g = 0.175
delta12 = deltas2[0]
delta21 = deltas1[1]
delta11 = deltas1[0]
delta22 = deltas2[1]

N = 100
CouplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)], [int((N + 1 + 2 * dx) / 2) , int((N + 1) / 2)]]
nT, maxT = 1000, 100

# Construct two photon GA object TGA and compute the eigenvalues/vectors of the Hamiltonian
giantAtom = TGA(N, J, nAtoms, [[g, g], [g, g]], deltas2, deltas1, CouplePoints)
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
