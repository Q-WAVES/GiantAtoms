import sys
import os

sys.path.append(os.path.abspath("../../src"))

from src.GA import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import numpy as np

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

#Fig 2c and d from DC=True, DFI=True
#Fig 2e and f from DC=True, BOC=True
#Fig 2g and h from DC=True, DFI=False
#Fig 3c and d from DC=False, DFI=True
#Fig 3e and f from DC=False, BOC=True
#Fig 3g and h from DC=False, DFI=False


#DC = False
DC = False
#DFI = False
DFI = True
#BOC = False
BOC = True
maxT, nT = 1000, 500
ts = np.linspace(0, maxT, nT)

np.random.seed(12345)

if DC:
    if BOC:
        U = 10
        J = 1
        nAtoms = 2
        N = 199
        dx = 2
        gs = [[0.0, 0.0], [0.0, 0.0]]
        gsDC = [[0.04, 0.04], [0.04, 0.04]]
        couplingPoints = couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)],
                                         [int((N + 1 + 2 * dx) / 2), int((N + 1) / 2)]]

        deltas2 = [9, 9]
        delta1 = [0, 0]

        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, deltas2, delta1, couplingPoints)
        initialState = giantAtom.constructInitialState(atoms = [1, 0], cavities = [0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$n^{(1)}_%i$' % iter)
            iter += 1

        plt.xlabel(r'$tJ$')
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.ylabel(r'$n^{(1)}(t)$')
        plt.xticks(np.linspace(0, maxT - 1, 3, dtype=int), np.linspace(0, maxT, 3, dtype=int))
        plt.title(r'$\Delta_1/J =  %.3f$' % (giantAtom.deltas2[0]))
        plt.tight_layout()
        #plt.savefig('../../Figures/PaperFigures/DCDFIBOC.svg')
        plt.show()

        heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState)

        maxPop = heatmapGG.max()

        fig, ax = plt.subplots(figsize=(7, 6))

        cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.01), aspect='auto')
        ax.invert_yaxis()

        numTicksY = 3
        numTicksX = 3
        tickPositionsY = np.linspace(0, giantAtom.N-1, numTicksY, dtype=int)
        tickLabelsY = np.linspace(1, giantAtom.N, numTicksY, dtype=int)
        tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
        tickLabelsX1 = ['0', '2.5e2', '5.0e2']
        tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

        ax.set_xticks(tickPositionsX, tickLabelsX)
        ax.set_yticks(tickPositionsY, tickLabelsY)
        ax.set_xlabel(r'$tJ$')
        ax.set_ylabel(r'Cavity index $(n)$')

        cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', location='top',
                             ticks=np.linspace(0, 0.01, 3))
        cbar2.set_label(r'$P(n, t)$', labelpad=10)
        fig.tight_layout()
        #fig.savefig('../../Figures/PaperFigures/DCDFIBOCHeatmap.svg')
        fig.show()

    elif DFI:
        U = 10
        J = 1
        nAtoms = 2
        N = 199
        dx = 2
        gs = [[0.0, 0.0], [0.0, 0.0]]
        gsDC = [[0.04, 0.04], [0.04, 0.04]]
        couplingPoints = couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)],
                                         [int((N + 1 + 2 * dx) / 2), int((N + 1) / 2)]]

        deltas2 = [10.392, 10.392]
        delta1 = [0, 0]

        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, deltas2, delta1, couplingPoints)
        initialState = giantAtom.constructInitialState(atoms = [1, 0], cavities = [0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$n^{(1)}_%i$' % iter)
            iter += 1

        plt.xlabel(r'$tJ$')
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.ylabel(r'$n^{(1)}(t)$')
        plt.xticks(np.linspace(0, maxT - 1, 3, dtype=int), np.linspace(0, maxT, 3, dtype=int))
        plt.title(r'$\Delta_1/J =  %.3f$' % (giantAtom.deltas2[0]))
        plt.tight_layout()
        #plt.savefig('../../Figures/PaperFigures/DCDFI.svg')
        plt.show()

        heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState)

        maxPop = heatmapGG.max()

        fig, ax = plt.subplots(figsize=(7, 6))

        cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.01), aspect='auto')
        ax.invert_yaxis()

        numTicksY = 3
        numTicksX = 3
        tickPositionsY = np.linspace(0, giantAtom.N-1, numTicksY, dtype=int)
        tickLabelsY = np.linspace(1, giantAtom.N, numTicksY, dtype=int)
        tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
        tickLabelsX1 = ['0', '2.5e2', '5.0e2']
        tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

        ax.set_xticks(tickPositionsX, tickLabelsX)
        ax.set_yticks(tickPositionsY, tickLabelsY)
        ax.set_xlabel(r'$tJ$')
        ax.set_ylabel(r'Cavity index $(n)$')

        cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', location='top',
                             ticks=np.linspace(0, 0.01, 3))
        cbar2.set_label(r'$P(n, t)$', labelpad=10)
        fig.tight_layout()
        #fig.savefig('../../Figures/PaperFigures/DCDFIHeatmap.svg')
        fig.show()

    else:
        U = 10
        J = 1
        nAtoms = 2
        N = 199
        dx = 2
        gs = [[0.0, 0.0], [0.0, 0.0]]
        gsDC = [[0.04, 0.04], [0.04, 0.04]]
        couplingPoints = couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)],
                                         [int((N + 1 + 2 * dx) / 2), int((N + 1) / 2)]]

        deltas2 = [10.5, 10.5]
        delta1 = [0, 0]

        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, deltas2, delta1, couplingPoints)
        initialState = giantAtom.constructInitialState(atoms=[1, 0], cavities=[0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$n^{(1)}_%i$' % iter)
            iter += 1

        plt.xlabel(r'$tJ$')
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.ylabel(r'$n^{(1)}(t)$')
        plt.title(r'$\Delta_1/J =  %.3f$' % (giantAtom.deltas2[0]))
        plt.xticks(np.linspace(0, maxT - 1, 3, dtype=int), np.linspace(0, maxT, 3, dtype=int))
        plt.tight_layout()
        #plt.savefig('../../Figures/PaperFigures/DCnoDFI.svg')
        plt.show()


        heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState)

        maxPop = heatmapGG.max()

        fig, ax = plt.subplots(figsize=(7, 6))

        cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.07), aspect='auto')
        ax.invert_yaxis()

        numTicksY = 3
        numTicksX = 3
        tickPositionsY = np.linspace(0, giantAtom.N-1, numTicksY, dtype=int)
        tickLabelsY = np.linspace(1, giantAtom.N, numTicksY, dtype=int)
        tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
        tickLabelsX1 = ['0', '2.5e2', '5.0e2']
        tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

        ax.set_xticks(tickPositionsX, tickLabelsX)
        ax.set_yticks(tickPositionsY, tickLabelsY)
        ax.set_xlabel(r'$tJ$')
        ax.set_ylabel(r'Cavity index $(n)$')

        cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', location='top',
                             ticks=np.linspace(0, 0.07, 3))
        cbar2.set_label(r'$P(n, t)$', labelpad=10)
        fig.tight_layout()
        #fig.savefig('../../Figures/PaperFigures/DCnoDFIHeatmap.svg')
        fig.show()
else:
    if BOC:
        U = 10
        J = 1
        nAtoms = 2
        N = 199
        dx = 2
        gs = [[0.25, 0.25], [0.25, 0.25]]
        gsDC = [[0.0, 0.0], [0.0, 0.0]]
        couplingPoints = couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)],
                                         [int((N + 1 + 2 * dx) / 2), int((N + 1) / 2)]]

        deltas2 = [7.0, 7.0]
        delta1 = [5.338, 5.338]

        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, deltas2, delta1, couplingPoints, frequencyDisorder=0.2, detuningDisorder=0.0)
        initialState = giantAtom.constructInitialState(atoms=[1, 0], cavities=[0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$n^{(2)}_%i$' % iter)
            iter += 1

        plt.xlabel(r'$tJ$')
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.ylabel(r'$n^{(2)}(t)$')
        plt.title(r'$\Delta_1/J =  %.3f, \Delta_2/J =  %.3f$' % (giantAtom.deltas1[0], giantAtom.deltas2[0]))
        plt.tight_layout()
        #plt.savefig('../../Figures/PaperFigures/noDCDFIBOC.svg')
        plt.show()


        heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState)

        maxPop = heatmapGG.max()

        fig, ax = plt.subplots(figsize=(7, 6))

        cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.04), aspect='auto')
        ax.invert_yaxis()

        numTicksY = 3
        numTicksX = 3
        tickPositionsY = np.linspace(0, giantAtom.N-1, numTicksY, dtype=int)
        tickLabelsY = np.linspace(1, giantAtom.N, numTicksY, dtype=int)
        tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
        tickLabelsX1 = ['0', '2.5e2', '5.0e2']
        tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

        ax.set_xticks(tickPositionsX, tickLabelsX)
        ax.set_yticks(tickPositionsY, tickLabelsY)
        ax.set_xlabel(r'$tJ$')
        ax.set_ylabel(r'Cavity index $(n)$')

        cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', location='top',
                             ticks=np.linspace(0, 0.04, 3))
        cbar2.set_label(r'$P(n, t)$', labelpad=10)
        fig.tight_layout()
        #fig.savefig('../../Figures/PaperFigures/noDCDFIBOCHeatmap.svg')
        fig.show()
    elif DFI:
        U = 10
        J = 1
        nAtoms = 2
        N = 199
        dx = 2
        gs = [[0.25, 0.25], [0.25, 0.25]]
        gsDC = [[0.0, 0.0], [0.0, 0.0]]
        couplingPoints = couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)],
                                         [int((N + 1 + 2 * dx) / 2), int((N + 1) / 2)]]

        deltas2 = [5.0, 5.0]
        delta1 = [5.338, 5.338]

        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, deltas2, delta1, couplingPoints, frequencyDisorder=0.2, detuningDisorder=0.0)
        initialState = giantAtom.constructInitialState(atoms=[1, 0], cavities=[0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$n^{(2)}_%i$' % iter)
            iter += 1

        plt.xlabel(r'$tJ$')
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.ylabel(r'$n^{(2)}(t)$')
        plt.title(r'$\Delta_1/J =  %.3f, \Delta_2/J =  %.3f$' % (giantAtom.deltas1[0], giantAtom.deltas2[0]))
        plt.tight_layout()
        #plt.savefig('../../Figures/PaperFigures/noDCDFI.svg')
        plt.show()


        heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState)

        maxPop = heatmapGG.max()

        fig, ax = plt.subplots(figsize=(7, 6))

        cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.04), aspect='auto')
        ax.invert_yaxis()

        numTicksY = 3
        numTicksX = 3
        tickPositionsY = np.linspace(0, giantAtom.N-1, numTicksY, dtype=int)
        tickLabelsY = np.linspace(1, giantAtom.N, numTicksY, dtype=int)
        tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
        tickLabelsX1 = ['0', '2.5e2', '5.0e2']
        tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

        ax.set_xticks(tickPositionsX, tickLabelsX)
        ax.set_yticks(tickPositionsY, tickLabelsY)
        ax.set_xlabel(r'$tJ$')
        ax.set_ylabel(r'Cavity index $(n)$')

        cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', location='top',
                             ticks=np.linspace(0, 0.04, 3))
        cbar2.set_label(r'$P(n, t)$', labelpad=10)
        fig.tight_layout()
        #fig.savefig('../../Figures/PaperFigures/noDCDFIHeatmap.svg')
        fig.show()

    else:
        U = 10
        J = 1
        nAtoms = 2
        N = 101
        dx = 2
        gs = [[0.04, 0.04], [0.04, 0.04]]
        gsDC = [[0.0, 0.0], [0.0, 0.0]]
        couplingPoints = couplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)],
                                         [int((N + 1 + 2 * dx) / 2), int((N + 1) / 2)]]

        deltas2 = [5.2, 5.2]
        delta1 = [5.338, 5.338]

        giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, deltas2, delta1, couplingPoints)
        initialState = giantAtom.constructInitialState(atoms=[1, 0], cavities=[0])
        Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)

        iter = 1
        for ns2 in Ns2:
            plt.plot(ts, ns2, linewidth=4, label=r'$n^{(2)}_%i$' % iter)
            iter += 1

        plt.xlabel(r'$tJ$')
        plt.legend(loc='upper right')
        plt.ylim(-0.1, 1.1)
        plt.ylabel(r'$n^{(2)}(t)$')
        plt.title(r'$\Delta_1/J =  %.3f, \Delta_2/J =  %.3f$' % (giantAtom.deltas1[0], giantAtom.deltas2[0]))
        plt.tight_layout()
        #plt.savefig('Figures/PaperFigures/noDCnoDFI.svg')
        plt.show()


        heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState)

        maxPop = heatmapGG.max()

        fig, ax = plt.subplots(figsize=(7, 6))

        cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.02), aspect='auto')
        ax.invert_yaxis()

        numTicksY = 3
        numTicksX = 3
        tickPositionsY = np.linspace(0, giantAtom.N-1, numTicksY, dtype=int)
        tickLabelsY = np.linspace(1, giantAtom.N, numTicksY, dtype=int)
        tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
        tickLabelsX1 = ['0', '2.5e2', '5.0e2']
        tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

        ax.set_xticks(tickPositionsX, tickLabelsX)
        ax.set_yticks(tickPositionsY, tickLabelsY)
        ax.set_xlabel(r'$tJ$')
        ax.set_ylabel(r'Cavity index $(n)$')

        cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', location='top',
                             ticks=np.linspace(0, 0.04, 3))
        cbar2.set_label(r'$P(n, t)$', labelpad=10)
        fig.tight_layout()
        #fig.savefig('Figures/PaperFigures/noDCnoDFI.svg')
        fig.show()
