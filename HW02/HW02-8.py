import numpy as np
from numpy import random

def sign(x):
    ret=np.ones(x.shape)
    for i, each in enumerate(x):
        if each<0: ret[i] = -1
    return ret

def getTheta(x):
    n = len(x)
    l1 = sorted(x)
    theta = np.zeros(n)
    for i in range(n-1):
        theta[i] = (l1[i]+l1[i+1])/2
    theta[-1] = 1
    return theta

def decision_stump():
    data_size = 2000
    iteration = 1000
    E_in = 0
    E_out = 0
    for i in range(iteration):
        x = random.uniform(-1 ,1 ,data_size)
        noise_rate = 0.2
        noise = sign(random.uniform(size = data_size) - noise_rate)
        y = sign(x) * noise
        theta = getTheta(x)
        e_in = np.zeros((2, data_size))
        for i in range(len(theta)):
            a1 = y * sign(x - theta[i])
            e_in[0][i] = (data_size - np.sum(a1))/(2 * data_size)
            e_in[1][i] = (data_size - np.sum(-a1))/(2 * data_size) 
        s = 0
        theta_best = 0
        min0, min1 = np.min(e_in[0]), np.min(e_in[1])
        if min0 < min1:
            s=1
            theta_best = theta[np.argmin(e_in[0])]
        else:
            s = -1
            theta_best = theta[np.argmin(e_in[1])]
        e_out = 0.5 + 0.3 * s * (np.abs(theta_best) - 1)
        E_in += np.min(e_in)
        E_out += np.min(e_out)
    ave_in = E_in / iteration
    ave_out = E_out / iteration
    print(f'{ave_in:.4f}', f'{ave_out:.4f}')

if __name__ == '__main__':
    decision_stump()
