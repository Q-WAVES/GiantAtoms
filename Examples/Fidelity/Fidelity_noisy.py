import numpy as np

import sys
import os

sys.path.append(os.path.abspath("../../src"))

from GA import GA_noisy, TGA_noisy


def compute_process_fidelity_vs_time(ts, N, J, g, delta, CouplePoints):

    zeta = 1.97
    d = 4
    Gamma_q=0.000016
    Gamma_c=0.00008

    # =================================================
    # SINGLE EXCITATION (GA) — diagonalize once
    # =================================================
    GA_obj = GA_noisy(
    N=N,
    J=J,
    nAtoms=2,
    gs=[[g, zeta*g, g], [g, zeta*g, g]],
    deltas=[-delta, delta],
    couplePoints=CouplePoints,
    Gamma_q=Gamma_q,
    Gamma_c=Gamma_c
    )
    GA_obj.computeEigens()

    psi_10 = GA_obj.constructInitialState(atom=1, cavity=0)
    psi_01 = GA_obj.constructInitialState(atom=2, cavity=0)

    GA_obj.computeDynamics(ts, psi_10)
    psi_10_t = GA_obj.finalStates   # shape (len(ts), dim)

    GA_obj.computeDynamics(ts, psi_01)
    psi_01_t = GA_obj.finalStates


    # =================================================
    # DOUBLE EXCITATION (TGA) — diagonalize once
    # =================================================
    TGA_obj = TGA_noisy(
        N=N,
        J=J,
        nAtoms=2,
        gs=[[g, zeta*g, g], [g, zeta*g, g]],
        deltas2=[-delta, -0.84],
        deltas1=[delta, -delta],
        couplePoints=CouplePoints,
        Gamma_q=Gamma_q,
        Gamma_c=Gamma_c
    )
    TGA_obj.computeEigens()

    psi_11 = TGA_obj.constructInitialState(atoms=[1,2], cavities=[0])

    TGA_obj.computeDynamics2_0(ts, psi_11)
    psi_11_t = TGA_obj.finalStates


    # =================================================
    # Helper functions
    # =================================================
    def build_env_dict_from_TGA(psi):
        env_dict = {}

        for k in range(len(psi)):
            atoms, cavities = TGA_obj.lookUpState(k)

            if len(atoms) == 2:
                a = 0
            elif len(atoms) == 1 and len(cavities) == 1:
                a = 1 if atoms[0] == 1 else 2
            elif len(atoms) == 0:
                a = 3
            else:
                continue

            w = tuple(cavities)

            if w not in env_dict:
                env_dict[w] = np.zeros(4, dtype=complex)

            env_dict[w][a] += psi[k]

        return env_dict


    def build_env_dict_from_GA(psi):
        env_dict = {}

        atomic_vec = np.zeros(4, dtype=complex)
        atomic_vec[1] = psi[0]
        atomic_vec[2] = psi[1]
        env_dict[()] = atomic_vec

        for n in range(len(psi) - 2):
            w = (n + 1,)
            atomic_vec = np.zeros(4, dtype=complex)
            atomic_vec[3] = psi[n + 2]
            env_dict[w] = atomic_vec

        return env_dict


    def build_env_dict_00():
        env_dict = {}
        vec = np.zeros(4, dtype=complex)
        vec[3] = 1.0
        env_dict[()] = vec
        return env_dict


    # =================================================
    # Ideal CZ Choi (constant)
    # =================================================
    U = np.diag([-1, 1, 1, 1])
    vecU = U.reshape(-1, order='F')
    J_target = np.outer(vecU, np.conj(vecU))


    # =================================================
    # Loop over times
    # =================================================
    F_list = []

    for ti in range(len(ts)):

        env_11 = build_env_dict_from_TGA(psi_11_t[ti])
        env_10 = build_env_dict_from_GA(psi_10_t[ti])
        env_01 = build_env_dict_from_GA(psi_01_t[ti])
        env_00 = build_env_dict_00()

        basis_env = {
            0: env_11,
            1: env_10,
            2: env_01,
            3: env_00
        }

        J_sim = np.zeros((d*d, d*d), dtype=complex)

        for i in range(d):
            for j in range(d):

                rho = np.zeros((d, d), dtype=complex)

                env_i = basis_env[i]
                env_j = basis_env[j]

                for w in env_i.keys():
                    if w in env_j:
                        rho += np.outer(env_i[w], np.conj(env_j[w]))

                for a in range(d):
                    for b in range(d):
                        J_sim[i*d + a, j*d + b] = rho[a, b]

        F_process = np.real(np.trace(J_sim.conj().T @ J_target) / (d*d))
        F_list.append(F_process)

    return np.array(F_list)

#gs = np.linspace(0.01,0.1,10)
gs=np.array([0.09])
print(gs)
N = 100
J = 1
delta = 0.17
zeta = 1.97

CouplePoints = [
    [48,50,52],
    [49,51,53]
]

F_max_list = []
t_max_list = []

import matplotlib.pyplot as plt

for g in gs:

    # predicted scaling
    t_guess = 74 * (0.1/g)**2

    ts = np.linspace(0,t_guess+100,101)
    print(ts)

    F_vs_t = compute_process_fidelity_vs_time(
        ts,
        N=N,
        J=J,
        g=g,
        delta=delta,
        CouplePoints=CouplePoints
    )

    plt.figure()
    plt.plot(ts, F_vs_t)

    idx = np.argmax(F_vs_t)

    F_max_list.append(F_vs_t[idx])
    t_max_list.append(ts[idx])

F_max_list = np.array(F_max_list)
t_max_list = np.array(t_max_list)

# =========================
# Plot (single plot)
# =========================

plt.figure()

plt.plot(gs, F_max_list, marker='o')

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 24
})

plt.ylim(-0.05, 1.05)
plt.xlim(0, gs.max()*1.2)

plt.xticks([0, 0.05, 0.1], fontsize=24)
plt.yticks([0, 0.5, 1], fontsize=24)

plt.xlabel(r"$g$", fontsize=24)
plt.ylabel(r"$F_{\max}$", fontsize=24)

plt.tight_layout()
plt.savefig("Fmax_vs_g.png", dpi=300)
plt.show()

plt.figure()

plt.plot(gs, t_max_list, marker='o')

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 24
})

plt.xticks([0, 0.05, 0.1], fontsize=24)
plt.yticks(fontsize=24)

plt.xlabel(r"$g$", fontsize=24)
plt.ylabel(r"$t_{\max}$", fontsize=24)

plt.tight_layout()
plt.savefig("tmax_vs_g.png", dpi=300)
plt.show()

data = np.column_stack((gs, F_max_list, t_max_list))

np.savetxt(
    "g_sweep_results.txt",
    data,
    header="g    F_max    t_max",
    fmt="%.10e"
)
