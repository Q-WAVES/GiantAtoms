import numpy as np

import sys
import os

sys.path.append(os.path.abspath("../../src"))

from GA import GA, TGA


def compute_process_fidelity_vs_time(ts, N, J, g, delta, CouplePoints):

    zeta = 1.97
    d = 4

    # =================================================
    # SINGLE EXCITATION (GA) — diagonalize once
    # =================================================
    GA_obj = GA(
        N=N,
        J=J,
        nAtoms=2,
        gs=[[g, zeta*g, g], [g, zeta*g, g]],
        deltas=[-delta, delta],
        couplePoints=CouplePoints
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
    TGA_obj = TGA(
        N=N,
        J=J,
        nAtoms=2,
        gs=[[g, zeta*g, g], [g, zeta*g, g]],
        deltas2=[-delta, -0.84],
        deltas1=[delta, -delta],
        couplePoints=CouplePoints
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

ts = np.linspace(0, 100, 101)

F_vs_t = compute_process_fidelity_vs_time(
    ts,
    N=100,
    J=1,
    g=0.1,
    delta=0.17,
    CouplePoints=[
        [48,50,52],
        [49,51,53]
    ]
)

# =========================
# Plot (single plot, no custom colors)
# =========================

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ts, F_vs_t)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 24
})


# Axis limits
plt.ylim(-0.05, 1.05)
plt.xlim(-5, 405)

# Ticks
plt.xticks([0, 100, 200, 300, 400], fontsize=24)
plt.yticks([0, 0.5, 1], fontsize=24)

plt.tight_layout()
plt.savefig("CZ_fidelity_vs_time_g=0.05.png", dpi=300)
plt.show()

print("Maximum process fidelity:", np.max(F_vs_t))
print("Time of maximum fidelity:", ts[np.argmax(F_vs_t)])

data = np.column_stack((ts, F_vs_t))

np.savetxt(
    "CZ_fidelity_vs_time_g=0.05.txt",
    data,
    header="Time    Process_Fidelity",
    fmt="%.10f"
)

