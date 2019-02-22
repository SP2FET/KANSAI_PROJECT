"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math
import matplotlib.pyplot as plt

# Estimation parameter of EKF
Q = np.diag([0.1, 0.1, 0.1, math.radians(1.0), math.radians(1.0), 1.0])**2 #~~~~~~~~~~~~~~~~~~~~~~~~
R = np.diag([1.0, 1.0, 1.0])**2

DT = 0.000000001  # time tick [s]

show_animation = True

vmat = np.matrix(np.zeros((6, 1)))

data = np.loadtxt("preIMUdata.csv",  # 読み込みたいファイルのパス
                  dtype="float32",
                  delimiter=",",  # ファイルの区切り文字
                  skiprows=0,  # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(1, 2, 3, 4, 5, 6)  # 読み込みたい列番号
                  )

def observation(xd, R_pos_ob, input_imu_v_x, input_imu_v_y, input_imu_v_z, i):
    ###  input current data here ~ XD ###

    input_slam_x = 0
    input_slam_y = 0
    input_slam_z = 0

    input_imu_a_x = data[i*100, 3]
    input_imu_a_y = data[i*100, 4]
    input_imu_a_z = data[i*100, 5]

    input_imu_w_rol = data[i*100, 0]
    input_imu_w_pit = data[i*100, 1]
    input_imu_w_yaw = data[i*100, 2]


    input_imu_v_x += DT * input_imu_a_x
    input_imu_v_y += DT * input_imu_a_y
    input_imu_v_z += DT * input_imu_a_z

    # SLAMの位置   (SLAM)
    zx = input_slam_x
    zy = input_slam_y
    zz = input_slam_z
    z = np.matrix([zx, zy, zz])

    # IMUの速度と角速度     (IMU)
    ud1 = input_imu_v_x  #verocity x
    ud2 = input_imu_v_y  #verocity y
    ud3 = input_imu_v_z  #verocity z
    w_rol = input_imu_w_rol  #rol
    w_pit = input_imu_w_pit  #pit
    ud5 = input_imu_w_yaw  #yaw


    if (w_rol >= 0 and w_pit >= 0) or (w_rol < 0 and w_pit < 0):   #theta (include rol pit)
        ud4 = math.sqrt(w_rol ** 2 + w_pit ** 2)
    else:
        ud4 = -math.sqrt(w_rol ** 2 + w_pit ** 2)

    ud = np.matrix([ud1, ud2, ud3, ud4, ud5]).T

    xd = motion_model(xd, ud, R_pos_ob)

    return z, xd, ud, w_rol, w_pit, input_imu_v_x, input_imu_v_y, input_imu_v_z


def a_to_t(a, v):
    t = a / 2 * (DT ** 2) + v * DT
    return t

def wrot_to_rot(wrot):
    rot = wrot * DT
    return rot


def motion_model(x, u, R_pos):

    vmat[5, 0] = math.sqrt(u[0, 0]**2 + u[1, 0]**2 + u[2, 0]**2)

    for i in range(3):
        u[i, 0] = a_to_t(0, u[i, 0])

    F = np.matrix([[1.0, 0, 0, 0, 0, 0],
                   [0, 1.0, 0, 0, 0, 0],
                   [0, 0, 1.0, 0, 0, 0],
                   [0, 0, 0, 1.0, 0, 0],
                   [0, 0, 0, 0, 1.0, 0],
                   [0, 0, 0, 0, 0, 0]])

    B = np.matrix([[R_pos[0, 0], R_pos[0, 1], R_pos[0, 2], 0, 0, 0],
                   [R_pos[1, 0], R_pos[1, 1], R_pos[1, 2], 0, 0, 0],
                   [R_pos[2, 0], R_pos[2, 1], R_pos[2, 2], 0, 0, 0],
                   [0, 0, 0, DT, 0, 0],
                   [0, 0, 0, 0, DT, 0]])    #??????????

    x = F * x + (u.T * B).T + vmat

    return x


def observation_model(x):
    #  Observation Model
    H = np.matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    z = H * x

    return z


def jacobF(x, vmat):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    theta = x[3, 0]
    yaw = x[4, 0]  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = vmat[5, 0]
    jF = np.matrix([[1.0, 0, 0, v*DT*math.cos(theta)*math.cos(yaw), -v*DT*math.sin(theta)*math.sin(yaw), math.sin(theta)*math.cos(yaw)],
                   [0, 1.0, 0, v*DT*math.sin(theta)*math.cos(yaw), v*DT*math.cos(theta)*math.sin(yaw), math.sin(theta)*math.sin(yaw)],
                   [0, 0, 1.0, 0.0, math.cos(theta), -v*DT*math.sin(theta)],
                   [0, 0, 0, 1.0, 0, 0],
                   [0, 0, 0, 0, 1.0, 0],
                   [0, 0, 0, 0, 0, 1.0]])

        #?????????????

    return jF


def jacobH(x):
    # Jacobian of Observation Model
    jH = np.matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u, R_pos):

    #  Predict
    xPred = motion_model(xEst, u, R_pos)
    jF = jacobF(xPred, vmat)
    PPred = jF * PEst * jF.T + Q

    #  Update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH * PPred * jH.T + R
    K = PPred * jH.T * np.linalg.inv(S)
    xEst = xPred + K * y
    PEst = (np.eye(len(xEst)) - K * jH) * PPred

    return xEst, PEst


def main():

    print(__file__ + " start!!")

    time = 0.0
    i = 0

    fig = plt.figure()
    ax = Axes3D(fig)

    R_pos_ob = np.matrix([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    input_imu_v_x = 0
    input_imu_v_y = 0
    input_imu_v_z = 0

    # State Vector [x y z theta yaw v]
    xEst = np.matrix(np.zeros((6, 1)))
    # xTrue = np.matrix(np.zeros((6, 1)))
    PEst = np.eye(6)

    xDR = np.matrix(np.zeros((6, 1)))  # Dead reckoning


    hxEst_x = []
    hxEst_y = []
    hxEst_z = []
    hxDR_x = []
    hxDR_y = []
    hxDR_z = []
    hz_x = []
    hz_y = []
    hz_z = []

    hxEst_x.append(xEst[0, 0])
    hxEst_y.append(xEst[1, 0])
    hxEst_z.append(xEst[2, 0])
    hxDR_x.append(xDR[0, 0])
    hxDR_y.append(xDR[1, 0])
    hxDR_z.append(xDR[2, 0])
    hz_x.append(0)
    hz_y.append(0)
    hz_z.append(0)

    while True:
        time += DT
        i += 1

        z, xDR, ud, w_rol_ob, w_pit_ob, input_imu_v_x, input_imu_v_y, input_imu_v_z= observation(xDR, R_pos_ob, input_imu_v_x, input_imu_v_y, input_imu_v_z, i)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud, R_pos_ob)

        rol_ob = wrot_to_rot(w_rol_ob)
        pit_ob = wrot_to_rot(w_pit_ob)
        w_yaw_ob = ud[4, 0]
        yaw_ob = wrot_to_rot(w_yaw_ob)

        R_ob = np.matrix([[math.cos(pit_ob) * math.cos(yaw_ob), -math.cos(pit_ob) * math.sin(yaw_ob), math.sin(pit_ob)],
                       [math.cos(rol_ob) * math.sin(yaw_ob) + math.sin(rol_ob) * math.sin(pit_ob) * math.cos(yaw_ob),
                        math.cos(rol_ob) * math.cos(yaw_ob) - math.sin(rol_ob) * math.sin(pit_ob) * math.sin(yaw_ob),
                        - math.sin(rol_ob) * math.cos(pit_ob)],
                       [math.sin(rol_ob) * math.sin(yaw_ob) - math.cos(rol_ob) * math.sin(pit_ob) * math.cos(yaw_ob),
                        math.sin(rol_ob) * math.cos(yaw_ob) + math.cos(rol_ob) * math.sin(pit_ob) * math.sin(yaw_ob),
                        math.cos(rol_ob) * math.cos(pit_ob)]])

        R_pos_ob = R_ob * R_pos_ob

        hxEst_x.append(xEst[0, 0])
        hxEst_y.append(xEst[1, 0])
        hxEst_z.append(xEst[2, 0])
        hxDR_x.append(xDR[0, 0])
        hxDR_y.append(xDR[1, 0])
        hxDR_z.append(xDR[2, 0])
        hz_x.append(z[0, 0])
        hz_y.append(z[0, 1])
        hz_z.append(z[0, 2])

        if show_animation:


            plt.cla()

            plt.title("t = " + str(i))

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.scatter(hz_x, hz_y, hz_z, color="gray", s=10, label="observed GPS")
            plt.plot(hxDR_x, hxDR_y, hxDR_z, color="red", marker="+", label="observed IMU")
            plt.plot(hxEst_x, hxEst_y, hxEst_z, color="blue", marker="", label="estimated")

            # plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.legend(loc='upper left', borderaxespad=0)
            plt.grid(True)
            plt.pause(0.001)

if __name__ == '__main__':
    main()