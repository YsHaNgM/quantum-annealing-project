from qutip import *
from scipy import *
import numpy as np
import matplotlib.pyplot as plt


N = 2


si = qeye(2)
sx = sigmax()
sz = sigmaz()

sx_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
    op_list.append(si)
