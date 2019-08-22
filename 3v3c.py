from qutip import *
from scipy import *
from random import *

import numpy as np
import matplotlib.pyplot as plt

# number of qubits
N = 9
M = 50
# vertex numbr
V = 3
# colour number
C = 3

# edges matrix
E = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
E = np.array(E)


# coefficients
h = 1.0
Jz = 1.0
Jx = 2.0

# time steps
taumax = 100.0
taulist = np.linspace(0, taumax, 100)

# variables in spin
si = qeye(2)
sx = sigmax()
sz = sigmaz()


si_list = []
sx_list = []
sz_list = []


for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))

    op_list[n] = si
    si_list.append(tensor(op_list))

# basis(0) for |-1| spin down 1 for |+1| spin up

# x(0) for sz |-1| spin down x(1) for sz |+1| spin up
# x = (sz+1)/2   sz=(-1) => x=0 sz=(+1) => x=1


# H0 transverse term
H0 = 0
for n in range(N):
    H0 += Jx * sx_list[n]

# initial state
ev, es = H0.eigenstates(eigvals=M)
psi_list = es[0]
psi0 = tensor(psi_list)


# Hp problem term
Hp = 0
vertex_sum = 0
for v in range(V):

    colour_sum = 0

    for c in range(C):
        colour_sum = colour_sum + sz_list[c + v * C]
    vertex_sum += (si_list[0] - colour_sum)**2
Hp += h * vertex_sum


colour_sum = 0
for c in range(C):
    # interaction terms
    for v in range(V):
        for v1 in range(v + 1, V):
            colour_sum += np.triu(E)[v][v1] * sz_list[c + v * C] * sz_list[c + v1 * C]
Hp += Jz * colour_sum


# the time-dependent hamiltonian in list-function format
args = {'t_max': max(taulist)}

h_t = [[H0, lambda t, args: (args['t_max'] - t) / args['t_max']],
       [Hp, lambda t, args: t / args['t_max']]]

evals_mat = np.zeros((len(taulist), M))
ekets_mat = np.zeros((len(taulist), M), dtype=object)

idx = [0]


def process_rho(tau, psi):

    # evaluate the Hamiltonian with gradually switched on interaction
    H = Qobj.evaluate(h_t, tau, args)

    # find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0], :] = real(evals)
    ekets_mat[idx[0], :] = ekets

    idx[0] += 1


# main evolution
sesolve(h_t, psi0, taulist, process_rho, args, options=Options(nsteps=100000), _safe_mode=True)

# print energies and configurations
print(evals_mat[len(taulist) - 1])

spin = 0

for config in range(6):
    print("configuration " + repr(config + 1) + ' ', end='')
    for num in range(9):
        spin = ekets_mat[len(taulist) - 1][config].ptrace(num)
        if spin == basis(2, 0).proj():
            print(0, end='')
        elif spin == basis(2, 1).proj():
            print(1, end='')
        if num % 3 == 2:
            print(' ', end='')
    print("\n")


plt.figure(figsize=(12, 6))

#
# plot the energy eigenvalues
#
for n in range(len(evals_mat[0, :])):
    ls, lw = ('b', 2) if n == 0 else ('k', 0.1)
    plt.plot(taulist / max(taulist), evals_mat[:, n], ls, lw=lw)

plt.xlabel(r'$\tau$')
plt.ylabel('Eigenenergies')
plt.title("Energyspectrum (%d lowest values) of %d spins.\n " % (M, N))
plt.savefig("3v3c")
