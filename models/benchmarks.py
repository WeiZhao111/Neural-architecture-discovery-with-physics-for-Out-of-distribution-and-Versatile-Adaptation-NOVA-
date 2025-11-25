import numpy as np
import matplotlib.pyplot as plt



def ZDT1(x, n):
    f1 = x[:, 0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[:, 1:], axis=1)
    f2 = g * (1 - np.power((f1 / g), 0.5))
    y = np.hstack([f1.reshape(-1, 1), f2.reshape(-1, 1)])
    return y


def ZDT2(x, n):
    f1 = x[:, 0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[:, 1:], axis=1)
    f2 = g * (1 - np.power((f1 * 1.0 / g), 2))
    y = np.hstack([f1.reshape(-1, 1), f2.reshape(-1, 1)])
    return y


def ZDT3(x, n):
    f1 = x[:, 0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[:, 1:], axis=1)
    f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))
    y = np.hstack([f1.reshape(-1, 1), f2.reshape(-1, 1)])
    return y


def ZDT4(x, n):
    f1 = x[:, 0]
    g = 1.0
    g += 10 * (n - 1)
    for i in range(1, 10):
        g += x[:, i] * x[:, i] - 10.0 * np.cos(4.0 * np.pi * x[:, i])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    y = np.hstack([f1.reshape(-1, 1), f2.reshape(-1, 1)])
    return y


def ZDT6(x, n):
    f1 = 1 - np.exp(-4 * x[:, 0]) * np.power(np.sin(6 * np.pi * x[:, 0]), 6)
    g = 1 + 9.0 * np.power(np.sum(x[:, 1:], axis=1) / (10 - 1.0), 0.25)
    f2 = g * (1 - np.power(f1 / g, 2))
    y = np.hstack([f1.reshape(-1, 1), f2.reshape(-1, 1)])
    return y



def ZDT1_LF(x, n):
    f1 = x[:, 0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[:, 1:], axis=1)
    h = 1 - np.power((f1 / g), 0.5)
    f2_LF = (0.8*g - 0.2)*(1.2*h + 0.2)
    y_LF = np.hstack([f1.reshape(-1, 1), f2_LF.reshape(-1, 1)])
    return y_LF


def ZDT2_LF(x, n):
    f1 = x[:, 0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[:, 1:], axis=1)
    h = 1 - np.power((f1 * 1.0 / g), 2)
    f2_LF = (0.9*g + 1.1) * (1.1*h - 0.1)
    y_LF = np.hstack([f1.reshape(-1, 1), f2_LF.reshape(-1, 1)])
    return y_LF


def ZDT3_LF(x, n):
    f1 = x[:, 0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[:, 1:], axis=1)
    h = 1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1)
    f2_LF = (0.75*g - 0.25) * (1.25*h + 0.25)
    y_LF = np.hstack([f1.reshape(-1, 1), f2_LF.reshape(-1, 1)])
    return y_LF



if __name__ == "__main__":
    x = 0.5*np.ones((10, 30))
    y = ZDT4(x)
    a