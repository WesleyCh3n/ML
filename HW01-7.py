import numpy as np
import matplotlib.pyplot as plt

class pocketPLA(object):
	"""This is the PLA pocket version class, 
	which user need to input the data path, the dimension of x features, and random seed
	Build in function include preprocess the x input and y output,
	also the PLA algorithm in iteration,
	the verification model to test the error
	"""
	def __init__(self, path, xDim, randomSeed):
		self.path = path
		self.Dim = xDim
		self.randomSeed = randomSeed

	def preprocess(self, path):
		data = np.loadtxt(path, dtype = float)
		if self.randomSeed == True : np.random.shuffle(data) 
		else: pass
		x = np.c_[np.ones((data.shape[0],1),dtype = float), data[:,:self.Dim]]
		y = data[:,self.Dim]
		return x, y

	def iteration(self):
		x, y = self.preprocess(self.path)
		w = np.zeros((self.Dim+1,1))
		wPk = np.zeros((self.Dim+1,1))
		smallErrorCount = len(x)
		update = 0
		while(update < 100):
			for i in range(len(x)):
				if np.dot(x[i], w)*y[i] <= 0:
					w += 0.5*(x[i]*y[i]).reshape(self.Dim+1,1)
					update += 1
					if self.verification(x, y, w) < self.verification(x, y, wPk):
						wPk = np.copy(w)
		return wPk

	def verification(self, x, y, w):
		errorCount = 0
		for i in range(len(x)):
			if np.dot(x[i], w)*y[i] <= 0:
				errorCount += 1
		return errorCount/len(x)

if __name__ == "__main__":	
	result = pocketPLA('hw1_7_train.dat', 4, True)
	testX, testY = result.preprocess('hw1_7_test.dat')

	print("Iteration Start. Please wait...")
	count = []
	for i in range(1126):
		w = result.iteration()
		print(f'{i}: Iteration Finish')
		count.append(result.verification(testX, testY, w))
	print("Finish")
	print(f"Average error rate is {sum(count)/1126.0}")

	plt.hist(count)
	plt.xlabel('Error Rate')
	plt.ylabel('Frequency')
	plt.show()
