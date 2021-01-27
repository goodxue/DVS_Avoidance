from allinc import * #导入所有库

def Kalman(Q,R,xmk_1,xk_1,Pmk_1):
	inv = np.linalg.inv
	deltat = 1e-3 #1ms
	A = np.array([
		[1,0,0,deltat,0,0],
		[0,1,0,0,deltat,0],
		[0,0,1,0,0,deltat],
		[0,0,0,1,0,0],
		[0,0,0,0,1,0],
		[0,0,0,0,0,1]
		])
	H = np.array([
		[1,0,0,0,0,0],
		[0,1,0,0,0,0],
		[0,0,1,0,0,0]
		])
	Q = 1 #未知
	R = 2 #未知
	v = np.random.normal(loc=0, scale=Q)
	w = np.random.normal(loc=0, scale=R)
	I = np.array([
		[1,0,0,0,0,0],
		[0,1,0,0,0,0],
		[0,0,1,0,0,0],
		[0,0,0,1,0,0],
		[0,0,0,0,1,0],
		[0,0,0,0,0,1]
		])
	xmk_1 = np.array([0,0,0,0,0,0]).T #未知
	xk_1 = np.array([0,0,0,0,0,0]).T #未知
	Pmk_1 = np.random.normal(loc=0, scale=1, size =6*6).reshape((6,6)) #未知
	a = time.time()
	it = 1000
	for i in range(it):
		xk = np.dot(A,xk_1) + v
		zk = np.dot(H,xk)+w
		xpk = np.dot(A,xmk_1)
		Ppk = A.dot(Pmk_1).dot(A.T)+Q
		Kk = Ppk.dot(H.T).dot(inv(H.dot(Ppk).dot(H.T)+R))
		xmk = xpk + Kk.dot(zk - H.dot(xpk))
		Pmk = (I - Kk.dot(H)).dot(Ppk).dot((I-Kk.dot(H)).T) + Kk.dot(R).dot(Kk.T)

		xmk_1 = xmk
		xk_1 = xk
		Pmk_1 = Pmk

if __name__ == '__main__':
	#Kalman(1,2)
	print((time.time()-a)*1000/it)