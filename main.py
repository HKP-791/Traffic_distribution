import numpy as np
import pandas as pd
import argparse

from algorithm.regression import gradient_descent

def Logarithmization(current_od, current_time):

    X = []
    y = []
    for i in range(len(current_O)):
        for j in range(len(current_D)):
            ln_O = np.log(current_O[i])
            ln_D = np.log(current_D[j])
            ln_C = np.log(current_time[i][j])
            ln_q = np.log(current_od[i][j])
            X.append([1, ln_O, ln_D, ln_C])  # 添加常数列1用于计算截距
            y.append(ln_q)

    X = np.array(X)
    y = np.array(y)

    return X, y


def unconstrained_gravity(future_O, future_D, future_time, k, alpha, beta, gamma):

    q_future = np.zeros((5, 5))
    for i in range(len(future_O)):
        for j in range(len(future_D)):
            O_i = future_O[i]
            D_j = future_D[j]
            c_ij = future_time[i, j]
            q = k * (O_i**alpha)*(D_j**beta) / (c_ij ** gamma)
            q_future[i, j] = q
            
    return q_future


def average_growth(q_future, future_O, future_D):

    iter_O = np.sum(q_future, axis=1)
    iter_D = np.sum(q_future, axis=0)
    for i in range(5):
        for j in range(5):
            O_i = future_O[i]
            D_j = future_D[j]
            Fo = O_i/iter_O[i]
            Fd = D_j/iter_D[j]
            q_future[i,j] = q_future[i, j]*(Fo+Fd)/2
        
    return q_future

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_od_path', type=str, default='data/current_od.csv', help='data path')
    parser.add_argument('--future_od_path', type=str, default='data/future_od.csv', help='data path')
    parser.add_argument('--current_time_path', type=str, default='data/current_time.csv', help='data path')
    parser.add_argument('--future_time_path', type=str, default='data/future_time.csv', help='data path')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=1000, help='number of iterations')
    args = parser.parse_args()

    # 初始化参数
    current_od = pd.read_csv(args.current_od_path, header=None).values
    current_time = pd.read_csv(args.current_time_path, header=None).values
    future_od = pd.read_csv(args.future_od_path, header=None).values
    future_time = pd.read_csv(args.future_time_path, header=None).values
    learning_rate = args.learning_rate
    num_iterations = args.num_iterations

    # 计算现状交通区的产生量和吸引量
    current_O = np.array([sum(row) for row in current_od])
    current_D = np.array([sum(col) for col in zip(*current_od)])
    future_O = future_od[0]
    future_D = future_od[1]

    X, y = Logarithmization(current_od, current_time)
    theta = gradient_descent(X, y, learning_rate, num_iterations)

    # 提取参数
    ln_k, alpha, beta, gamma_negative = theta
    k = np.exp(ln_k)
    gamma = -gamma_negative

    print("无约束重力模型参数：\n", f"k = {k:.4f}, α = {alpha:.4f}, β = {beta:.4f}, γ = {gamma:.4f}\n")

    q_future = unconstrained_gravity(future_O, future_D, future_time, k, alpha, beta, gamma)
    fo = np.abs(1-(future_O/np.sum(q_future, axis=1)))
    fd = np.abs(1-(future_D/np.sum(q_future, axis=0)))
    e = np.max(np.concatenate((fo, fd)))

    while e > 0.003:
        current_od = q_future
        q_future = average_growth(current_od, future_O, future_D)

        fo = np.abs(1-(future_O/np.sum(q_future, axis=1)))
        fd = np.abs(1-(future_D/np.sum(q_future, axis=0)))
        e = np.max(np.concatenate((fo, fd)))

    print("未来交通分布预测结果矩阵：\n", pd.DataFrame(q_future, 
                                      columns=[1,2,3,4,5], 
                                      index=[1,2,3,4,5]).to_string(float_format='%.0f'))
