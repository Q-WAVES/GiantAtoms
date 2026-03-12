import sys, os

from src.GA import TGADoublon

sys.path.append(os.path.abspath("../../src"))

from GA import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import comb
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

plt.rcParams['font.size'] = 40
plt.rcParams['figure.figsize'] = [7, 7]
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'times',
    'text.latex.preamble': r'''
        \usepackage{times}
        \usepackage{amsmath,amssymb,bm}
        \DeclareSymbolFont{letters}{OML}{cmm}{m}{it}
    '''
})

# Fig 2a inset:       BS = "2D", delta1 = [0], delta2 = [10.392], gs = [[0, 0], [0, 0]], gsDC = [[0.04, 0.04], [0.04, 0.04]]
# Fig 2a main figure: BS = "1D", delta1 = [0], delta2 = [10.392], gs = [[0, 0], [0, 0]], gsDC = [[0.04, 0.04], [0.04, 0.04]]

# Fig 3a inset:       BS = "2D", delta1 = [5.337], delta2 = [5.0], gs = [[0.25, 0.25], [0.25, 0.25]], gsDC = [[0, 0], [0, 0]]
# Fig 3a main figure: BS = "1D", delta1 = [5.337], delta2 = [5.0], gs = [[0.25, 0.25], [0.25, 0.25]], gsDC = [[0, 0], [0, 0]]

BS = '1D'

def DFIfreq(k, U, J):
    return np.sqrt(U ** 2 + 16 * J ** 2 * np.cos(k / 2) ** 2)


nAtoms = 1
N = 199
dx = 2
couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)]]
delta1 = [5.337]
delta2 = [5.0]
gs = [[0.25, 0.25], [0.25, 0.25]]
gsDC = [[0.0, 0.0], [0.0, 0.0]]
J = 1
U = 10

giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, delta2, delta1, couplePoints)
#giantAtom = TGADoublon.loadGA('SavedGAs/1TGADoublondx2N199g025Delta1_5_337Delta2_5.joblib')

es, vs = giantAtom.computeEigens()
IPR = giantAtom.computeIPR()
indeces = np.linspace(0, len(giantAtom.Hamiltonian) - 1, len(giantAtom.Hamiltonian))

maxIndex = np.argmax(IPR)

DBStates = []
eigenEnergies = []

for i in range(0):
    IPR = np.delete(IPR, maxIndex)

    vs = np.delete(vs, maxIndex, axis=1)
    es = np.delete(es, maxIndex)
    maxIndex = np.argmax(IPR)


if BS == '2D':
    plt.rcParams['font.size'] = 40
    for i in range(1):
        doublonState = np.zeros((giantAtom.N, giantAtom.N))

        localizedEigenState = vs[:, maxIndex]
        twoPhotonEigenState = localizedEigenState[comb(giantAtom.nAtoms + 1, 2) + giantAtom.N * giantAtom.nAtoms:]
        eigenEnergy = es[maxIndex]

        for j in range(giantAtom.N):
            for k in range(j, giantAtom.N):
                index = j * giantAtom.N - j * (j - 1) / 2 + k - j
                doublonState[j, k] = np.abs(twoPhotonEigenState[int(index)]) ** 2
                doublonState[k, j] = np.abs(twoPhotonEigenState[int(index)]) ** 2

        DBStates.append(doublonState)
        eigenEnergies.append(eigenEnergy)

        IPR = np.delete(IPR, maxIndex)

        vs = np.delete(vs, maxIndex, axis=1)
        es = np.delete(es, maxIndex)
        maxIndex = np.argmax(IPR)

elif BS == '1D':
    plt.rcParams['figure.figsize'] = [7, 6]
    plt.rcParams['font.size'] = 28
    for i in range(1):
        doublonState = np.zeros(giantAtom.N)

        localizedEigenState = vs[:, maxIndex]
        twoPhotonEigenState = localizedEigenState[comb(giantAtom.nAtoms + 1, 2) + giantAtom.N * giantAtom.nAtoms:]
        eigenEnergy = es[maxIndex]

        for j in range(giantAtom.N):
            index = j * giantAtom.N - j * (j - 1) / 2
            doublonState[j] = np.abs(twoPhotonEigenState[int(index)]) ** 2

        DBStates.append(doublonState)
        eigenEnergies.append(eigenEnergy)

        IPR = np.delete(IPR, maxIndex)

        vs = np.delete(vs, maxIndex, axis=1)
        es = np.delete(es, maxIndex)
        maxIndex = np.argmax(IPR)

iter = 0

if DBStates[0].ndim == 1:
    for DBState in DBStates:
        eigenEnergy = eigenEnergies[iter]

        plt.plot(DBState, linewidth=4)
        plt.xlabel(r'Cavity index $(i)$')
        plt.xticks(np.linspace(0, giantAtom.N - 1, 5, dtype=int), np.linspace(1, giantAtom.N, 5, dtype=int))
        # plt.ylim([0 - 0.00005, 0.0008 + 0.00005])
        #plt.yticks(np.linspace(0, 0.0006, 5))
        plt.ylabel(r'$P_b(n, n)$')
        plt.xlabel(r'Cavity index ($n$)')
        plt.title(r'$E/J = %.3f$' % eigenEnergy)
        plt.tight_layout()
        plt.show()
        iter += 1
elif DBStates[0].ndim == 2:
    for DBState in DBStates:

        fig, ax = plt.subplots()

        eigenEnergy = eigenEnergies[iter]
        cax = ax.imshow(DBState[89:110, 89:110], cmap = 'hot', vmin = 0, vmax = 0.025)
        ax.invert_yaxis()
        ax.set(xticks=np.linspace(0, 20, 2, dtype=int), yticks=np.linspace(0, 20, 2, dtype=int),
               xticklabels=np.linspace(90, 110, 2, dtype=int), yticklabels=np.linspace(90, 110, 2, dtype=int))
        ax.set_ylabel(r'$n$')
        ax.set_xlabel(r'$m$')
        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', location='top', ticks=np.linspace(0, 0.025, 2))
        cbar.set_label(r'$P_b(m, n)$', labelpad=10)
        fig.tight_layout()
        fig.show()
        iter += 1

