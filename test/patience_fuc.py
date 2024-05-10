import numpy as np


def func1(x):
    return -np.log(x) + 3


def func2(x):
    return 0.5 / x


def func3(x):
    return -0.05 * x ** 2 + 5


def normalize(func, x_range):
    min_val = func(x_range[0])
    max_val = func(x_range[0])

    for x in x_range:
        val = func(x)
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

    # 归一化函数
    normalized_func = lambda x: (func(x) - min_val) / (max_val - min_val)
    return normalized_func


# 设定 x 的范围
x_range = np.linspace(0.1, 10, 1000)  # 0.1 到 10 之间的 1000 个点

# 归一化函数
moderate_patience = normalize(func1, x_range)
low_patience = normalize(func2, x_range)
high_patience = normalize(func3, x_range)
