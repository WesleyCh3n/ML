import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    zback = 1/(1+np.exp(-1*z))
    return zback

# Logistic Regression
def logisticReg(X, Y, eta, numiter, flag=0):
    row, col = X.shape
    theta = np.zeros((col, 1))
    num = 0
    for i in range(numiter):
        if flag == 0:
            derr = (-1*X*Y).T.dot(sigmoid(-1*X.dot(theta)*Y))/row
        else:
            if num >= row:
                num = 0
            derr = -Y[num, 0]*X[num: num+1, :].T*sigmoid(-1*X[num, :].dot(theta)[0]*Y[num, 0])
            num += 1
        theta -= eta*derr
    return theta

def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.values
    col, row = data.shape
    X = np.c_[np.ones((col, 1)), data[:, 0: row-1]]
    Y = data[:, row-1:row]
    return X, Y

def mistake(X, Y, theta):
    yhat = X.dot(theta)
    yhat[yhat > 0] = 1
    yhat[yhat <= 0] = -1
    err = np.sum(yhat != Y)/len(Y)
    return err


X, Y = loadData('hw3_train.dat')
Xtest, Ytest = loadData('hw3_test.dat')

grd = []
sgrd = []

n = 0.01 # modify this
for i in range(2000):
    theta = logisticReg(X, Y, n, i, 0)
    errout = mistake(Xtest, Ytest, theta)
    grd.append(errout)
    print('Eout = ', errout)
    theta = logisticReg(X, Y, n, i, 1)
    errout = mistake(Xtest, Ytest, theta)
    sgrd.append(errout)
    print('Eout = ', errout)

plt.scatter(range(2000), sgrd, color = 'blue', s = 1)
plt.scatter(range(2000), grd, color = 'red', s = 1)
plt.show()
