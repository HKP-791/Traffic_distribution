import numpy as np

def gradient_descent(X, y, learning_rate, num_iterations):
    theta = np.zeros(X.shape[1])  # [ln_k, alpha, beta, gamma_negative]
    m = len(y)

    # 迭代更新参数
    for _ in range(num_iterations):
        y_pred = X @ theta
        error = y_pred - y
        gradient = (X.T @ error) / m
        theta -= learning_rate * gradient
        if _ % 100 == 0:
            print(f"迭代次数：{_}, 损失函数值：{np.mean((y_pred - y) ** 2)}")

    return theta