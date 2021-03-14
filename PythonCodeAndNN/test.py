import numpy as np 

if __name__ == '__main__':
	x1 = np.array([[0,1,1],[1,1,1]]).T
	x2 = np.array([[0,1,1],[1,1,1]]).T
	print(np.cov(x1,x2))