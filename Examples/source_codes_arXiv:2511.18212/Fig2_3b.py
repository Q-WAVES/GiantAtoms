import sys, os

sys.path.append(os.path.abspath("../../src"))

from GA import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

plt.rcParams['font.size'] = 28
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

# Fig 2b DC = True, 3b DC = False

#DC = False
DC = True
maxT, nT = 100, 1000
ts = np.linspace(0, maxT, nT)


if DC:
    deltas2 = [[10.392], [10.5], [9.0]]
    for i in range(3):
        N = 199
        dx = 2
        couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)]]
        J = 1
        U = 10
        nAtoms = 1
        gs = [[0.0, 0.0], [0.0, 0.0]]
        gsDC = [[0.04, 0.04], [0.04, 0.04]]
        delta1 = [0]
        delta2 = deltas2[i]
        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, delta2, delta1, couplePoints)
        #giantAtom = TGADoublon.loadGA('SavedGAs/1TGADoublondx2N199g004Delta2_10_5' + str(i) + '.joblib')
        initialState = giantAtom.constructInitialState(atoms = [1, 0], cavities = [0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$\Delta_1 / J = %.3f$' % giantAtom.deltas2[0])
            iter += 1

    plt.xlabel(r'$tJ$')
    plt.legend(loc='lower left')
    plt.ylim(0.49, 1.01)
    plt.ylabel(r'$n^{(1)}(t)$')
    plt.tight_layout()
    #plt.savefig('Figures/PaperFigures/1GACompDC.svg')
    plt.show()
else:
    deltas2 = [[5.0], [5.2], [7.0]]
    for i in range(3):
        N = 199
        dx = 2
        couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)]]
        J = 1
        U = 10
        nAtoms = 1
        gs = [[0.25, 0.25], [0.25, 0.25]]
        gsDC = [[0.0, 0.0], [0.0, 0.0]]
        delta1 = [5.338]
        delta2 = deltas2[i]
        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, delta2, delta1, couplePoints)
        #giantAtom = TGADoublon.loadGA('SavedGAs/TGADoublondx2N199g025Delta1_5_6Delta2_5' + str(i) + '.joblib')
        initialState = giantAtom.constructInitialState(atoms = [1, 0], cavities = [0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$\Delta_2 /J= %.3f$' % giantAtom.deltas2[0])
            iter += 1

    plt.xlabel(r'$tJ$')
    plt.legend(loc='lower left')
    plt.ylim(0.49, 1.01)
    plt.ylabel(r'$n^{(2)}(t)$')
    plt.title(r'$\Delta_1 /J=  %.3f$' % (giantAtom.deltas1[0]))
    plt.tight_layout()
    #plt.savefig('Figures/PaperFigures/1GACompg0_25noDC.svg')
    plt.show()


