# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as ran
import random
import matplotlib.pyplot as plt

# generate norm
def norm(_loc=0.0, _scale=1.0, _size=(1)):
    return ran.normal(_loc, _scale, _size)

print("a")

####################################################################
###########################  初期値の設定  ##########################
####################################################################
flag = True

# 時刻
global_time = 0
# t+1の刻み設定
dt=0.1
# 計算回数
calc_num = 1000
end_time = calc_num * dt
# 正規分布の発生に関するパラメータ
mean_a = 0
sigma_a = 3
mean_z = 0
sigma_z = 10

def noise():
    # noise = random.uniform(-5.0, 5.0)
    noise = ran.normal(loc = mean_z, scale = sigma_z, size = (1))
    return noise

def accel():
    # accel = random.uniform(-5.0, 5.0)
    accel = ran.normal(loc = mean_a, scale = sigma_a, size = (1))
    return accel

first = random.uniform(-30.0, 30.0)
first_er = norm(mean_z, sigma_z)

print("初期真値 : "+str(first))
print("初期誤差 : "+str(first_er))
print("初期和 : "+str(first + first_er))

# 単位行列
I = np.matrix([[1 , 0],
               [0, 1]])
# Plot用
ground_truth_position=[]
observed_position=[]
estimate_position=[]
time_series=[]

# ground_truth_position.append(first)
# observed_position.append(first_er)

##########       状態方程式        ##########
# トロッコの位置と速度: [位置, 加速度]
x_k = np.matrix([
                [first_er],
                [0]
            ])

x_k_true = np.matrix([
                [first],
                [0]
            ])
x_k_k = x_k_true + x_k

# 運動方程式 [位置+(加速度*時間), 加速度]
F = np.matrix([
                [1, dt],
                [0, 1]
            ])
# 時間遷移に関する雑音モデルの行列 (0平均かつQの正規分布に従う)
G = np.matrix([
                [(dt**2) / 2],
                [dt]
            ])

##########       観測方程式        ##########
# 誤差行列
p_k = np.matrix([
                [0, 0],
                [0, 0]
             ])
p_k_k = p_k
# 位置のみを線形写像する観測モデル
H = np.matrix([
                1,
                0
             ])
# cov(Gw_k) = (sigma_a)^2 * (G)(G^T): 共分散
Q = (sigma_a**2) * G * G.T
# R = E(v_k*(v_k)^t) = (sigma_z)^2: ?
R = sigma_z**2

v_k = first_er

####################################################################
###########################  計算スタート  ##########################
####################################################################
while global_time < end_time:

    ##########       観測        ##########

    # v_k: 観測誤差 (偶然誤差)
    if (flag == True):
        flag = False
    else:
        v_k = noise()
    # z_k = Hx_k + v_k: トロッコの位置をセンサで観測する
    z_k = H * x_k_true + v_k
    observed_position.append(z_k.tolist()[0][0])

    ##########       予測        ##########
    # w_k = [a_k]: トロッコの加速度(誤差混み)
    w_k = accel()
    # Fx_{k-1} + Gw_k: 現時刻における予測推定値
    x_k = (F * x_k_k) + (G * w_k)

    ## 真値(次の位置)
    # Fx_{k-1} + Gw_k: （補正しない）現時刻における予測推定値
    x_k_true = (F * x_k_true) + (G * w_k)
    ground_truth_position.append(x_k_true.tolist()[0][0])

    ##########     補正と更新     ##########
    # F * P_{k-1} * F^T + G_k * Q_k * (G_k)^T: 現時刻における予測誤差行列
    p_k = F * p_k_k * F.T + Q
    # R + H * P_k * H^T: 観測残差の共分散
    S_k = (H * p_k) * H.T + R
    # P_k * H^T * S^-1: 最適カルマンゲイン
    K_k = p_k * H.T * S_k.I
    # z_k - H * x_k: 観測残差
    e_k = z_k - H * x_k
    # x_k + K_k * e_k: 位置の補正
    if (flag == True):
        flag = False
    else:
        x_k_k = x_k + K_k * e_k
    estimate_position.append(x_k_k.tolist()[0][0])
    # (I - K_k * H) * p_k_k: 更新された誤差の共分散
    p_k_k = (I - K_k * H) * p_k



    ##########    タイムカウント    ##########
    time_series.append(global_time)
    global_time += dt


MSE = np.sum((np.array(ground_truth_position)-np.array(observed_position))**2)
print("観測誤差"+str(MSE))
MSE = np.sum((np.array(ground_truth_position)-np.array(estimate_position))**2)
print("カルマンフィルタによる推定誤差"+str(MSE))


plt.plot(time_series, observed_position, color="green", marker="", label="observed")
plt.plot(time_series, estimate_position, color="red", marker="", label="estimation")
plt.plot(time_series, ground_truth_position, color="blue", marker="", label="groundtruth")
plt.legend()
plt.show()