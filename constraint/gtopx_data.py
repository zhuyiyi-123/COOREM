import os
from ctypes import *
import numpy as np
import random


class ConstrainOfflineTask():

    def __init__(self, benchmark):
        self.benchmark = benchmark
        if benchmark == 1:
            self.obj_num = 1
            self.var_num = 6
            self.con_num = 4
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0]  # lower bounds
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0]  # upper bounds
        if benchmark == 2:
            self.obj_num = 1
            self.var_num = 22
            self.con_num = 0
            self.xl = [-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05,
                       1.05, 1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi]
            self.xu = [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5,
                       291.0, np.pi, np.pi, np.pi, np.pi]
        if benchmark == 3:
            self.obj_num = 1
            self.var_num = 18
            self.con_num = 0
            self.xl = [1000.0, 1.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.1, -np.pi,
                       -np.pi, -np.pi]
            self.xu = [4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99, 0.99, 0.99, 6.0, 6.0, 6.0, np.pi,
                       np.pi, np.pi]
        if benchmark == 4:
            self.obj_num = 1
            self.var_num = 26
            self.con_num = 0
            self.xl = [1900.0, 2.5, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                       0.01,
                       1.1, 1.1, 1.05, 1.05, 1.05, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
            self.xu = [2300.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 600.0, 0.99, 0.99, 0.99, 0.99, 0.99,
                       0.99,
                       6.0, 6.0, 6.0, 6.0, 6.0, np.pi, np.pi, np.pi, np.pi, np.pi]
        if benchmark == 5:
            self.obj_num = 1
            self.var_num = 8
            self.con_num = 6
            self.xl = [3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0]
            self.xu = [10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0]
        if benchmark == 6:
            self.obj_num = 1
            self.var_num = 22
            self.con_num = 0
            self.xl = [1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.06,
                       1.05, 1.05, 1.05, -np.pi, -np.pi, -np.pi, -np.pi]
            self.xu = [1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9, 0.9, 0.9, 0.9, 0.9, 9.0, 9.0,
                       9.0, 9.0, np.pi, np.pi, np.pi, np.pi]
        if benchmark == 7:
            self.obj_num = 1
            self.var_num = 12
            self.con_num = 2
            self.xl = [7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0, -np.pi, -np.pi]
            self.xu = [9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9, 0.9, 7.0, 500.0, np.pi, np.pi]
        if benchmark == 8:
            self.obj_num = 1
            self.var_num = 10
            self.con_num = 4
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0, 1.0, 1.0, 1.0, 1.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0, 9.0, 9.0, 9.0, 9.0]
        if benchmark == 9:
            self.obj_num = 2
            self.var_num = 6
            self.con_num = 5
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0]
        if benchmark == 10:
            self.obj_num = 2
            self.var_num = 10
            self.con_num = 5
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0, 1.0, 1.0, 1.0, 1.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0, 9.0, 9.0, 9.0, 9.0]
        self.f = [0.0] * self.obj_num
        self.g = [0.0] * self.con_num
        self.x_values = []
        self.xx = []
        self.x_mean = []
        self.x_std = []
        self.y_mean = 0
        self.y_std = 0
        self.cons_mean = []
        self.cons_std = []
        self.num = 2

    def prediction(self, x):
        if os.name == "posix":
            lib_name = "gtopx.so"  # Linux//Mac/Cygwin
        else:
            lib_name = "gtopx.dll"  # Windows
        lib_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib_name
        CLIB = CDLL(lib_path)
        f_ = (c_double * self.obj_num)()
        benchmark_ = c_long(self.benchmark)
        value = []
        constraint = []

        for j in range(0, len(x)):
            x_ = (c_double * self.var_num)()
            for i in range(0, self.var_num):
                x_[i] = c_double(x[j][i])
            if self.con_num > 0:
                g_ = (c_double * self.con_num)()
            if self.con_num == 0:
                g_ = (c_double * 1)()
            CLIB.gtopx(benchmark_, f_, g_, x_)
            for i in range(0, self.obj_num):
                self.f[i] = f_[i]
            for i in range(0, self.con_num):
                self.g[i] = g_[i]
            value.append(self.f[0])
            constraint.append(self.g[:])
        return value, constraint

    def random_data(self, num):
        # print(self.xl, self.xu)
        # random_nums = [
        #     random.uniform(np.array([-980.,  10.,  95. , 25.  , 395. ,  995. , -4. , -4. , -4. , -4.]),np.array([5.,  405.,
        #                     475.,  405., 2005., 6005. , 14.,  14.,  14. , 14.]))
        #     for _ in range(num)]
        random_nums = [
            random.uniform((np.array(np.array(self.xu) + np.ones((1, self.var_num))[0]) * 50),np.array((np.array(self.xl) - np.ones((1, self.var_num))) * 50)[0])
            for _ in range(num)]
        num = 0
        # print(random_nums)
        num_true = 0
        y = np.zeros((len(random_nums), 1))
        cons = -10000 * np.ones((len(random_nums), self.con_num))
        for i in range(len(random_nums)):
            for j in range(len(random_nums[0])):
                if random_nums[i][j] > self.xu[j] or random_nums[i][j] < self.xl[j]:
                    num = num + 1
            if num != 0:
                y[i] = 10000
            else:
                y[i], q = self.prediction([list(random_nums[i])])
                cons[i] = q[0]
            num = 0
        return random_nums, y, cons

    @property
    def x(self):
        self.xx = []
        for i in range(self.num):
            for lower, upper in zip(self.xl, self.xu):
                x = random.uniform(lower, upper)
                self.x_values.append(x)
            self.xx.append(self.x_values)
            self.x_values = []
        return self.xx

    @property
    def y(self, x):
        y_truth, constraint = self.prediction(self, x)
        return y_truth, constraint

    def dataset(self, num):
        self.num = num
        x = self.x
        y, g = self.prediction(x=x)
        return np.array(x), np.array(y), np.array(g)

    def normalize_x(self, x):
        x = np.array(x)
        standardized_x = []
        for i in range(len(x)):
            x_mean = np.mean(x[i])
            x_std = np.std(x[i])
            standardized = (x[i] - x_mean) / x_std
            standardized_x.append(standardized.tolist())
            self.x_mean.append(x_mean)
            self.x_std.append(x_std)
        return standardized_x

    def normalize_y(self, y):
        y = np.array(y)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        standardized = (y - self.y_mean) / self.y_std
        return standardized
    
    def normalize_cons(self, cons):
        cons = np.array(cons)
        standardized_cons = []
        for i in range(len(cons)):
            cons_mean = np.mean(cons[i])
            cons_std = np.std(cons[i])
            standardized = (cons[i] - cons_mean) / cons_std
            standardized_cons.append(standardized.tolist())
            self.cons_mean.append(cons_mean)
            self.cons_std.append(cons_std)
            print(self.cons_mean, self.cons_std)
        return standardized_cons

    def denormalize_x(self, x):
        origin_x = []
        ori_x = np.zeros((len(x[0]), 1))
        for i in range(len(x)):
            for j in range(len(x[0])):
                ori_x[j] = x[i][j] * self.x_std[i].astype(np.float64) + self.x_mean[i]
            origin_x.append(ori_x)
        return origin_x

    def denormalize_y(self, y):
        origin_y = y * self.y_std + self.y_mean
        return origin_y
    
    def denormalize_cons(self, cons):
        origin_cons = []
        ori_cons = np.zeros((len(cons), len(cons[0])))
        print(self.cons_std, self.cons_mean)
        ori_cons = cons * self.cons_std + self.cons_mean
        # for i in range(len(cons)):
        #     for j in range(len(cons[0])):
        #         print(i,j)
        #         print(cons[i][j], )
        #         ori_cons[i][j] = cons[i][j] * self.cons_std[i].astype(np.float64) + self.cons_mean[i]
        #     origin_cons.append(ori_cons)
        return origin_cons
