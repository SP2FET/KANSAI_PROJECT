import numpy as np
import numpy.random as rd
import pandas as pd

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns

import random

sns.set(style="whitegrid", palette="muted", color_codes=True)

N = 200

a = -2
b = -1

n_particle = 10**3 * 5
sigma_2 = 2**a
alpha_2 = 10**b

# 正規分布の発生に関するパラメータ
mean_a = 0
sigma_a = 1
mean_z = 0
sigma_z = 1

def noise():
    # noise = random.uniform(low = -5.0, high = 5.0)
    # noise = rd.normal(loc = mean_z, scale = sigma_z, size = (1))
    noise = rd.triangular(left = -10, mode = 0, right = 10)
    return noise

def accel():
    # accel = random.uniform(-5.0, 5.0)
    accel = rd.normal(loc = mean_a, scale = sigma_a, size = (10))
    # accel = triangular(left, mode, right[, size])
    return accel


def make_pandas_data(n):

    dt = 0.1

    first = random.uniform(-5.0, 5.0)

    ground_truth_position = []
    observed_position = []

    # 運動方程式 [位置+(加速度*時間), 加速度]
    F = np.matrix([
        [1, dt],
        [0, 1]
    ])

    # 時間遷移に関する雑音モデルの行列 (0平均かつQの正規分布に従う)
    G = np.matrix([
        [(dt ** 2) / 2],
        [dt]
    ])

    # 位置のみを線形写像する観測モデル
    H = np.matrix([
        1,
        0
    ])

    x_k_true = np.matrix([
        [first],
        [0]
    ])

    for i in range(n):
        #observed position list
        # v_k : noise
        v_k = noise()
        z_k = H * x_k_true + v_k
        observed_position.append(z_k.tolist()[0][0])

        # w_k : inputed acceralation 入力された加速度
        w_k = accel()

        #real position list
        x_k_true = (F * x_k_true) + (G * w_k)
        ground_truth_position.append(x_k_true.tolist()[0][0])

    ground_truth_df = pd.DataFrame(ground_truth_position, columns = ["data"])
    observed_df = pd.DataFrame(observed_position, columns = ["data"])

    return ground_truth_df, observed_df


class ParticleFilter(object):
    def __init__(self, y, n_particle, sigma_2, alpha_2):
        self.y = y
        self.n_particle = n_particle
        self.sigma_2 = sigma_2
        self.alpha_2 = alpha_2
        self.log_likelihood = -np.inf

    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2 * np.pi * s2)) ** (-1) * np.exp(-(y - x) ** 2 / (2 * s2))

    def F_inv(self, w_cumsum, idx, u):
        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        return k + 1

    def resampling(self, weights):
        w_cumsum = np.cumsum(weights)
        idx = np.asanyarray(range(self.n_particle))
        k_list = np.zeros(self.n_particle, dtype=np.int32)  # サンプリングしたkのリスト格納場所

        # 一様分布から重みに応じてリサンプリングする添え字を取得
        for i, u in enumerate(rd.uniform(0, 1, size=self.n_particle)):
            k = self.F_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list

    def resampling2(self, weights):
        """
        計算量の少ない層化サンプリング
        """
        idx = np.asanyarray(range(self.n_particle))
        u0 = rd.uniform(0, 1 / self.n_particle)
        u = [1 / self.n_particle * i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(weights)
        k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
        return k

    def simulate(self, seed=71):
        rd.seed(seed)

        # 時系列データ数
        T = len(self.y)

        # 潜在変数
        x = np.zeros((T + 1, self.n_particle))
        x_resampled = np.zeros((T + 1, self.n_particle))


        # $1 Set particle first time
        # 潜在変数の初期値
        initial_x = rd.normal(0, 1, size=self.n_particle)
        x_resampled[0] = initial_x
        x[0] = initial_x

        # 重み
        w = np.zeros((T, self.n_particle))
        w_normed = np.zeros((T, self.n_particle))

        l = np.zeros(T)  # 時刻毎の尤度

        for t in range(T):
            print("\r calculating... t={}".format(t), end="")
            for i in range(self.n_particle):
                # $2 Set next particle with special noise
                # 1階差分トレンドを適用
                v = rd.normal(0, np.sqrt(self.alpha_2 * self.sigma_2))  # System Noise
                x[t + 1, i] = x_resampled[t, i] + v  # システムノイズの付加
                # $3 Calculate likelihood from y[n - 1] with dedicated Gaussian
                w[t, i] = self.norm_likelihood(self.y[t], x[t + 1, i], self.sigma_2)  # y[t]に対する各粒子の尤度
            # $4 Adjust for sum of weight gets to 1.0
            w_normed[t] = w[t] / np.sum(w[t])  # 規格化
            l[t] = np.log(np.sum(w[t]))  # 各時刻対数尤度

            # $5 Resampling
            # k = self.resampling(w_normed[t]) # リサンプルで取得した粒子の添字
            k = self.resampling2(w_normed[t])  # リサンプルで取得した粒子の添字（層化サンプリング）
            x_resampled[t + 1] = x[t + 1, k]

        # 全体の対数尤度
        self.log_likelihood = np.sum(l) - T * np.log(n_particle)

        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l

    # $6 load average
    def get_filtered_value(self):
        """
        尤度の重みで加重平均した値でフィルタリングされ値を算出
        """
        return np.diag(np.dot(self.w_normed, self.x[1:].T))

    def draw_graph(self, ground_truth_df):
        # グラフ描画
        T = len(self.y)

        plt.figure(figsize=(16, 8))
        plt.plot(range(T), self.y, color="b", label="observed")
        plt.plot(range(T), ground_truth_df, color="r", marker="", label="groundtruth")
        plt.plot(self.get_filtered_value(), "g", label="estimated")

        for t in range(T):
            plt.scatter(np.ones(self.n_particle) * t, self.x[t], color="lightgray", s=2, alpha=0.1)

        plt.title("sigma^2={0}, alpha^2={1}, log likelihood={2:.3f}".format(self.sigma_2,
                                                                            self.alpha_2,
                                                                            self.log_likelihood))
        plt.show()

def main():
    ground_truth_df, observed_df = make_pandas_data(N)

    pf = ParticleFilter(observed_df.data.values, n_particle, sigma_2, alpha_2)

    pf.simulate()

    pf.draw_graph(ground_truth_df)


if __name__ == "__main__": main()