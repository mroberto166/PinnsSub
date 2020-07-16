import os
import sys
import itertools
import subprocess
from ImportFile import *

equation = sys.argv[1]
cluster = sys.argv[2]
mode = sys.argv[3]

if equation == "poisson":
    nx = np.array([20, 40, 80, 160])
    ny = nx

    N_int = (0.5625 * nx * ny).astype(int)
    N_coll = (nx * ny).astype(int) - N_int
    N_u = np.zeros_like(N_coll).astype(int)
    folders = ["PoissH1_20", "PoissH1_40", "PoissH1_80", "PoissH1_160"]

elif equation == "heat":

    T = 0.02
    h = 1 / 50
    a = 0.2
    nx = np.array([int(1 / h), int(2 / h), int(4 / h)])
    nt = np.array([16, 16, 16])
    nx_int = ((1 - 2 * a) * nx).astype(int)
    nt_int = nt
    N_int = nx_int * nt_int
    N_coll = (nx * nt).astype(int) - N_int
    N_u = np.zeros_like(N_coll).astype(int)
    # folders = ["HeatH1_50_uni", "HeatH1_100_uni", "HeatH1_200_uni"]
    folders = ["HeatH1_50_T", "HeatH1_100_T", "HeatH1_200_T"]


elif equation == "wave":
    omega_1 = np.array([[0, 1],
                        [0., 0.2]])
    omega_2 = np.array([[0, 1],
                        [0.8, 1.0]])
    nx = np.array([30, 60, 90, 120])
    nt = nx
    print(float(omega_1[1, 1] - omega_1[1, 0]))
    print(float(omega_2[1, 1] - omega_2[1, 0]))
    if Ec.domain == "GC":
        nx_int = (2 * float(omega_1[1, 1] - omega_1[1, 0]) * nx).astype(int)
    else:
        nx_int = (float(omega_1[1, 1] - omega_1[1, 0]) * nx).astype(int)
    nt_int = nt
    N_int = nx_int * nt_int
    print(nx_int)
    N_coll = (nx * nt).astype(int) - N_int
    N_u = np.zeros_like(N_coll).astype(int)
    # folders = ["WaveH1_30_uni", "WaveH1_60_uni", "WaveH1_90_uni", "WaveH1_120_uni"]
    folders = ["WaveH1_30b", "WaveH1_60b", "WaveH1_90b", "WaveH1_120b"]

elif equation == "stokes":

    radius = 0.25
    A = 1
    A_omega = pi * radius ** 2
    nx = np.array([20, 40, 80, 160])
    # nx = np.array([40, 80, 160])
    ny = nx
    N_int = (A_omega / A * nx * ny).astype(int)
    N_coll = (nx * ny).astype(int) - N_int
    N_u = np.zeros_like(N_coll).astype(int)
    folders = ["StokesH1_20_b", "StokesH1_40_b", "StokesH1_80_b"]
else:
    raise ValueError()

for i in range(len(folders)):

    N_coll_set = N_coll[i]
    N_u_set = N_u[i]
    N_int_set = N_int[i]
    folder_name = folders[i]

    print("\n")
    print("##########################################")
    print("Number of samples:")
    print(" - Collocation points:", N_coll_set)
    print(" - Initial and boundary points:", N_u_set)
    print(" - Internal points:", N_int_set)
    print("\n")

    arguments = list()

    arguments.append(str(N_coll_set))
    arguments.append(str(N_u_set))
    arguments.append(str(N_int_set))
    arguments.append(str(folder_name))
    arguments.append(str(cluster))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            if mode == "ensemble":
                string_to_exec = "bsub -W 00:10 python3 EnsambleTraining.py "
            elif mode == "resampling":
                string_to_exec = "bsub -W 00:10 python3 SampleSenitivity.py "
            else:
                raise ValueError()
        else:
            if mode == "ensemble":
                string_to_exec = "python3 EnsambleTraining.py "
            elif mode == "resampling":
                string_to_exec = "python3 SampleSenitivity.py "
            else:
                raise ValueError()
        for arg in arguments:
            string_to_exec = string_to_exec + " " + str(arg)
        os.system(string_to_exec)
    else:
        python = os.environ['PYTHON36']
        p = subprocess.Popen([python, "EnsambleTraining.py"] + arguments)
        p.wait()
