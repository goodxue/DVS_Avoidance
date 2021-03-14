from allinc import * #导入所有库

def Kalman(xmk_1_,xk_1_,Q_=1,R_=1):
	inv = np.linalg.inv
	pinv = np.linalg.pinv
	deltat = 1e-2 #10ms
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
	Q = Q_ #未知
	R = R_ #未知
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
	xmk_1 = xmk_1_ #6
	xk_1 = xk_1_ #6
	x1 = xmk_1[:3]
	x2 = xk_1[:3]
	v1 = xmk_1[3:]
	v2 = xk_1[3:]
	Pmk_1 = np.cov(np.array([x1,v1]).T,np.array([x2,v2]).T) #6:6


	#迭代
	xk = np.dot(A,xk_1) + v
	zk = np.dot(H,xk)+w
	xpk = np.dot(A,xmk_1)
	Ppk = A.dot(Pmk_1).dot(A.T)+Q
	try:
		Kk = Ppk.dot(H.T).dot(inv(H.dot(Ppk).dot(H.T)+R))
	except:
		Kk = Ppk.dot(H.T).dot(pinv(H.dot(Ppk).dot(H.T)+R))


	xmk = xpk + Kk.dot(zk - H.dot(xpk))
	Pmk = (I - Kk.dot(H)).dot(Ppk).dot((I-Kk.dot(H)).T) + Kk.dot(R).dot(Kk.T)

	xmk_1 = xmk
	xk_1 = xk
	Pmk_1 = Pmk
	return xmk

if __name__ == '__main__':
	print(Kalman(np.array([0,1,1,0,1,0]).T,np.array([1,0,1,0,0,1]).T))
	#print((time.time()-a)*1000/it)