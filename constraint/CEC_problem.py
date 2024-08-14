import numpy as np


def cec20_func(x, prob_k=5, D=7):
    ps, D = x.shape
    global initial_flag
    global G, B, P, Q, L
    initial_flag = None
    G = None
    B = None
    P = None
    Q = None
    L = None

    if prob_k == 1:
        # Heat Exchanger Network Design (case 1)
        f = 35 * x[:, 0] ** 0.6 + 35 * x[:, 1] ** 0.6
        g = np.zeros(ps)
        h = np.zeros((ps, 8))
        h[:, 0] = 200 * x[:, 0] * x[:, 3] - x[:, 2]
        h[:, 1] = 200 * x[:, 1] * x[:, 5] - x[:, 4]
        h[:, 2] = x[:, 2] - 10000 * (x[:, 6] - 100)
        h[:, 3] = x[:, 4] - 10000 * (300 - x[:, 6])
        h[:, 4] = x[:, 2] - 10000 * (600 - x[:, 7])
        h[:, 5] = x[:, 4] - 10000 * (900 - x[:, 8])
        h[:, 6] = (x[:, 3] * np.log(np.abs(x[:, 7] - 100) + 1e-8) - x[:, 3] * np.log(600 - x[:, 6] + 1e-8) - x[:, 7] +
                   x[:, 6] + 500)
        h[:, 7] = x[:, 5] * np.log(np.abs(x[:, 8] - x[:, 6]) + 1e-8) - x[:, 5] * np.log(600) - x[:, 8] + x[:, 6] + 600

    elif prob_k == 2:
        # Heat Exchanger Network Design (case 2)
        f = (x[:, 0] / (120 * x[:, 3])) ** 0.6 + (x[:, 1] / (80 * x[:, 4])) ** 0.6 + (x[:, 2] / (40 * x[:, 5])) ** 0.6
        g = np.zeros(ps)
        h = np.zeros((ps, 9))
        h[:, 0] = x[:, 0] - 1e4 * (x[:, 7] - 100)
        h[:, 1] = x[:, 1] - 1e4 * (x[:, 8] - x[:, 7])
        h[:, 2] = x[:, 2] - 1e4 * (500 - x[:, 8])
        h[:, 3] = x[:, 0] - 1e4 * (300 - x[:, 9])
        h[:, 4] = x[:, 1] - 1e4 * (400 - x[:, 10])
        h[:, 5] = x[:, 2] - 1e4 * (600 - x[:, 11])
        h[:, 6] = (x[:, 3] * np.log(np.abs(x[:, 9] - 100) + 1e-8) - x[:, 3] * np.log(300 - x[:, 7] + 1e-8) - x[:, 9] -
                   x[:, 7] + 400)
        h[:, 7] = x[:, 4] * np.log(np.abs(x[:, 10] - x[:, 7]) + 1e-8) - x[:, 4] * np.log(
            np.abs(400 - x[:, 8]) + 1e-8) - x[:, 10] + x[:, 7] - x[:, 8] + 400
        h[:, 8] = x[:, 5] * np.log(np.abs(x[:, 11] - x[:, 8]) + 1e-8) - x[:, 5] * np.log(100) - x[:, 11] + x[:, 8] + 100

    elif prob_k == 3:
        f = 1.715 * x[:, 0] + 0.035 * x[:, 0] * x[:, 5] + 4.0565 * x[:, 2] + 10.0 * x[:, 1] - 0.063 * x[:, 2] * x[:, 4]
        h = np.zeros(ps)
        g = np.zeros((ps, 14))
        g[:, 0] = (0.0059553571 * x[:, 5] * x[:, 5] * x[:, 0] + 0.88392857 * x[:, 2] - 0.1175625 * x[:, 5] * x[:, 0]
                   - x[:, 0])
        g[:, 1] = 1.1088 * x[:, 0] + 0.1303533 * x[:, 0] * x[:, 5] - 0.0066033 * x[:, 0] * x[:, 5] * x[:, 5] - x[:, 2]
        g[:, 2] = (6.66173269 * x[:, 5] * x[:, 5] + 172.39878 * x[:, 4] - 56.596669 * x[:, 3] - 191.20592 * x[:, 5]
                   - 10000)
        g[:, 3] = 1.08702 * x[:, 5] + 0.32175 * x[:, 3] - 0.03762 * x[:, 5] * x[:, 5] - x[:, 4] + 56.85075
        g[:, 4] = (0.006198 * x[:, 6] * x[:, 3] * x[:, 2] + 2462.3121 * x[:, 1] - 25.125634 * x[:, 1] * x[:, 3]
                   - x[:, 2] * x[:, 3])
        g[:, 5] = (161.18996 * x[:, 2] * x[:, 3] + 5000.0 * x[:, 1] * x[:, 3] - 489510.0 * x[:, 1] - x[:, 2] * x[:, 3]
                   * x[:, 6])
        g[:, 6] = 0.33 * x[:, 6] - x[:, 4] + 44.333333
        g[:, 7] = 0.022556 * x[:, 4] - 0.007595 * x[:, 6] - 1.0
        g[:, 8] = 0.00061 * x[:, 2] - 0.0005 * x[:, 0] - 1.0
        g[:, 9] = 0.819672 * x[:, 0] - x[:, 2] + 0.819672
        g[:, 10] = 24500.0 * x[:, 1] - 250.0 * x[:, 1] * x[:, 3] - x[:, 2] * x[:, 3]
        g[:, 11] = 1020.4082 * x[:, 3] * x[:, 1] + 1.2244898 * x[:, 2] * x[:, 3] - 100000. * x[:, 1]
        g[:, 12] = 6.25 * x[:, 0] * x[:, 5] + 6.25 * x[:, 0] - 7.625 * x[:, 2] - 100000
        g[:, 13] = 1.22 * x[:, 2] - x[:, 5] * x[:, 0] - x[:, 0] + 1.0
    elif prob_k == 4:
        x[:, 2] = np.round(x[:, 2])
        g = np.zeros((ps, 3))
        f = -0.7 * x[:, 2] + 5 * (x[:, 0] - 0.5) ** 2 + 0.8
        g[:, 0] = -np.exp(x[:, 0] - 0.2) - x[:, 1]
        g[:, 1] = x[:, 1] + 1.1 * x[:, 2] + 1
        g[:, 2] = x[:, 0] - x[:, 2] - 0.2
        g[:, 3] = x[:, 0] - 0.2
        g[:, 4] = 1 - x[:, 0]
        g[:, 5] = x[:, 1] + 2.22554
        g[:, 6] = -1 - x[:, 1]
        h = np.zeros(ps)
    elif prob_k == 5:
        f = (2 * np.sqrt(2) * x[:, 0] + x[:, 1]) * 100
        g = np.zeros((ps, 7))
        g[:, 0] = -(x[:, 1] / (np.sqrt(2) * x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1]) * 2 - 2)
        g[:, 1] = -((np.sqrt(2) * x[:, 0] + x[:, 1]) / (np.sqrt(2) * x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1]) * 2 - 2)
        g[:, 2] = -(1 / (np.sqrt(2) * x[:, 1] + x[:, 0]) * 2 - 2)
        g[:, 3] = x[:, 0]
        g[:, 4] = 1 - x[:, 0]
        g[:, 5] = x[:, 1]
        g[:, 6] = 1 - x[:, 1]
    elif prob_k == 6:
        x[:, 1] = np.round(x[:, 1])
        f = x[:, 1] + 2 * x[:, 0]
        g = np.zeros((ps, 6))
        g[:, 0] = -(-x[:, 0] ** 2 - x[:, 1] + 1.25)
        g[:, 1] = -(x[:, 0] + x[:, 1] - 1.6)
        g[:, 2] = x[:, 0]
        g[:, 3] = 1.6 - x[:, 0]
        g[:, 4] = x[:, 1]
        g[:, 5] = 1 - x[:, 1]
    elif prob_k == 19:
        f = 1.10471 * x[:,0]**2 * x[:,1] + 0.04811 * x[:,2] * x[:,3] * (14 + x[:,1])
        P = 6000
        L = 14
        delta_max = 0.25
        E = 30 * 1e6
        G = 12 * 1e6
        T_max = 13600
        sigma_max = 30000
        Pc = 4.013 * E * np.sqrt(x[:,2]**2 * x[:,3]**6 / 30) / L**2 * (1 - x[:,2] / (2 * L) * np.sqrt(E / (4 * G)))
        sigma = 6 * P * L / (x[:,3] * x[:,2]**2)
        delta = 6 * P * L**3 / (E * x[:,2]**2 * x[:,3])
        J = 2 * (np.sqrt(2) * x[:,0] * x[:,1] * (x[:,1]**2 / 4 + (x[:,0] + x[:,2])**2 / 4))
        R = np.sqrt(x[:,1]**2 / 4 + (x[:,0] + x[:,2])**2 / 4)
        M = P * (L + x[:,1] / 2)
        ttt = M * R / J
        tt = P / (np.sqrt(2) * x[:,0] * x[:,1])
        t = np.sqrt(tt**2 + 2 * tt * ttt * x[:,1] / (2 * R) + ttt**2)
        # constraints
        g = np.zeros((x.shape[0], 13))
        g[:,0] = -(x[:,0] - x[:,3])
        g[:,1] = -(sigma - sigma_max)
        g[:,2] = -(P - Pc)
        g[:,3] = -(t - T_max)
        g[:,4] = -(delta - delta_max)
        g[:,5] = x[:,0] - 0.125
        g[:,6] = 2 - x[:,0]
        g[:,7] = x[:,1] - 0.1
        g[:,8] = 10 - x[:,1]
        g[:,9] = x[:,2] - 0.1
        g[:,10] = 10 - x[:,2]
        g[:,11]= x[:,3]-0.1
        g[:,12] = 2-x[:,3]
    return f, g