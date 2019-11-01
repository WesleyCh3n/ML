import numpy as np
import matplotlib.pyplot as plt

class PLA(object):
	"""This is the PLA class, 
	which user need to input the data path, the dimension of x features, and random seed
	Build in function include preprocess the x input and y output,
	also the PLA algorithm in iteration. 
	"""
	def __init__(self, path, xDim, randomSeed):
		self.path = path
		self.Dim = xDim
		self.randomSeed = randomSeed

	def preprocess(self):
		data = np.loadtxt(self.path, dtype = float)
		if self.randomSeed == True : np.random.shuffle(data) 
		else: pass
		x = np.c_[np.ones((data.shape[0],1),dtype = float), data[:,:self.Dim]]
		y = data[:,self.Dim]
		return x,y

	def iteration(self):
		x, y = self.preprocess()
		w = np.zeros((self.Dim+1,1))
		update = 0
		while(True):
			finish = True
			for i in range(len(x)):
				if np.dot(x[i], w)*y[i] <= 0:
					w += (x[i]*y[i]).reshape(self.Dim+1,1)
					update += 1
					finish = False
			if finish == True:
				break
		return update

if __name__ == "__main__":
	
	result = PLA('hw1_15_train.dat', 4, True)
	
	print("Iteration Start. Please wait...")
	count = []
	for i in range(1126):
		count.append(result.iteration())
		print(f'{i}: Iteration Finish')
	print("Finish")

	print(f"Average number of update is {sum(count)/1126}")
	plt.hist(count)
	plt.xlabel('Number of updates')
	plt.ylabel('Frequency of number')
	plt.show()
	