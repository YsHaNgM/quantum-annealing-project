
# import resource

import sys
import threading

from qutip import *
from scipy import *
from random import *
import numpy as np
import matplotlib.pyplot as plt

sys.setrecursionlimit(100000)
threading.stack_size(536870912 * 2)
# sys.setrecursionlimit(10**6)

# target_stack = 2**29
# cur_stack, max_stack = resource.getrlimit(resource.RLIMIT_STACK)
# target_stack = min(max_stack, target_stack)
# resource.setrlimit(resource.RLIMIT_STACK, (max(cur_stack, target_stack), max_stack))

N = 12
M = 6
# vertex numbr
# S E W N stands for countries of UK
V = 4
# colour number
C = 3

# edges matrix
E = [[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
E = np.array(E)


h = 1.0 * np.ones(V)  # 1.0 * 2 * pi * (1 - 2 * np.ones(N))
Jz = 2 * np.ones(N)  # abs(1.0 * 2 * pi * (1 - 2 * rand(N)))
Jx = 1.0 * 2 * pi * (1 - 2 * rand(N))

# Jy = 1.0 * 2 * pi * (1 - 2 * rand(N))

taumax = 100.0
taulist = np.linspace(0, taumax, 100)

si = qeye(2)
sx = sigmax()
sz = sigmaz()
# sy = sigmay()

si_list = []
sx_list = []
sz_list = []
# sy_list = []

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

#     op_list[n] = sy
#     sy_list.append(tensor(op_list))

# basis(0) for |-1| spin down 1 for |+1| spin up

# x(0) for sz |-1| spin down x(1) for sz |+1| spin up
# x = (sz+1)/2   sz=(-1) => x=0 sz=(+1) => x=1

psi_list = [basis(2, 0), basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1), basis(
    2, 0), basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 0)]
# for n in range(N):
#     psi_list.append(basis(2,randrange(1)))
# psi_list = [basis(2,0) for n in range(N)]
psi0 = tensor(psi_list)
# H0 transverse term
H0 = 0
for n in range(N):
    H0 += - 0.5 * 2.5 * sx_list[n]


# Hp problem term
Hp = 0
vertex_sum = 0
for v in range(V):
    colour_sum = 0
    for c in range(C):
        colour_sum += sz_list[c + v * C]
    vertex_sum += (sz_list[0] - colour_sum)**2
Hp += h[v] * vertex_sum
# for n in range(N):
#     Hp += h[n] * sz_list[n]


for v in range(V):
    # interaction terms
    vertex_sum = 0
    for v1 in range(v + 1, V):

        colour_sum = 0
        for c in range(C):
            colour_sum += sz_list[c + v * C] * sz_list[c + v1 * C]
        vertex_sum += np.triu(E)[v][v1] * colour_sum
    Hp += h[v] * vertex_sum
#     Hp += Jz[n] * sz_list[n] * sz_list[(n+1)]
#     H1 += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]


# the time-dependent hamiltonian in list-function format
args = {'t_max': max(taulist)}

h_t = [[H0, lambda t, args: (args['t_max'] - t) / args['t_max']],
       [Hp, lambda t, args: t / args['t_max']]]

evals_mat = np.zeros((len(taulist), M))
P_mat = np.zeros((len(taulist), M))

idx = [0]


def process_rho(tau, psi):

    # evaluate the Hamiltonian with gradually switched on interaction
    H = Qobj.evaluate(h_t, tau, args)

    # find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0], :] = real(evals)

    # find the overlap between the eigenstates and psi
    for n, eket in enumerate(ekets):
        P_mat[idx[0], n] = abs((eket.dag().data * psi.data)[0, 0])**2

    idx[0] += 1


def process(h_t, psi0, taulist, process_rho, args):
    mesolve(h_t, psi0, taulist, [], process_rho, args)


print("start")

t = threading.Thread(target=process(h_t, psi0, taulist, process_rho, args))
t.start()

t.join()
t.run()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

#
# plot the energy eigenvalues
#

# first draw thin lines outlining the energy spectrum
for n in range(len(evals_mat[0, :])):
    ls, lw = ('b', 1) if n == 0 else ('k', 0.25)
    axes[0].plot(taulist / max(taulist), evals_mat[:, n] / (2 * pi), ls, lw=lw)

# second, draw line that encode the occupation probability of each state in
# its linewidth. thicker line => high occupation probability.
for idx in range(len(taulist) - 1):
    for n in range(len(P_mat[0, :])):
        lw = 0.5 + 4 * P_mat[idx, n]
        if lw > 0.55:
            axes[0].plot(array([taulist[idx], taulist[idx + 1]]) / taumax,
                         array(
                             [evals_mat[idx, n], evals_mat[idx + 1, n]]) / (2 * pi),
                         'r', linewidth=lw)

axes[0].set_xlabel(r'$\tau$')
axes[0].set_ylabel('Eigenenergies')
axes[0].set_title("Energyspectrum (%d lowest values) of a chain of %d spins.\n " % (M, N)
                  + "The occupation probabilities are encoded in the red line widths.")

#
# plot the occupation probabilities for the few lowest eigenstates
#
for n in range(len(P_mat[0, :])):
    if n == 0:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n], 'r', linewidth=2)
    else:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n])

axes[1].set_xlabel(r'$\tau$')
axes[1].set_ylabel('Occupation probability')
axes[1].set_title("Occupation probability of the %d lowest " % M +
                  "eigenstates for a chain of %d spins" % N)
axes[1].legend(("Ground state",))
