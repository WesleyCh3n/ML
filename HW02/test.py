import numpy as np 
import random

def genData(N,noise):
    x = []
    y = []
    for i in range(N):
        x1 = random.uniform(-1,1)
        prob = random.uniform(0,1)
        if prob < noise:
            y1 = -sign(x1,True)
        else:
            y1 = sign(x1,True)
        x.append(x1)
        y.append(y1)
    return np.array(x),np.array(y)

def errorRate(x,y,s,theta,h):
    errorNum = 0
    for i in range(len(x)):
        if y[i] != h(x[i],s,theta):
            errorNum += 1
    return errorNum/len(x)

def hFunc(x,s,theta):
    if s:
        return sign(x-theta,s)
    else:
        return -sign(x-theta,s)

def sign(v,s):
    if v == 0:
        return s
    elif v < 0:
        return -1
    else:
        return 1

def trainDecisionStump(N,noise):
    x,y = genData(N,noise) #N=20,noise=0.2
    E_in = 1
    best_s = True
    best_theta = 0
    thetas = np.sort(x) #排序x用來作為備選的theta
    ss = [True,False] #先遍歷s是正的時候
    for theta in thetas:
        for s in ss:
            E = errorRate(x,y,s,theta,hFunc)
            if E < E_in:
                E_in = E
                best_s = s
                best_theta = theta
    index, = np.where(thetas == best_theta)
    if index[0] == 0:
        best_theta = (-1+best_theta)/2
    else:
        best_theta = (thetas[index[0]-1]+best_theta)/2
    E_out = computeEout(best_s,best_theta)
    return E_in,E_out

def computeEout(s,theta):
    if s:
        return 0.5+0.3*(np.abs(theta)-1)
    else:
        return 0.5-0.3*(np.abs(theta)-1)

def main():
    iteration = 1000
    N = 2000
    noise = 0.2
    err_in_sum = 0
    err_out_sum = 0
    for i in range(iteration):
        err_in, err_out = trainDecisionStump(N,noise)
        err_in_sum += err_in
        err_out_sum += err_out 
        if i%100 == 99:
            print("iteration: ",i+1)
    print("total errorRate in sample is",err_in_sum/iteration)
    print("total errorRate out of sample is",err_out_sum/iteration)

if __name__ == '__main__':
    main()
