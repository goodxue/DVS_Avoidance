from allinc import *

def D2toD3(u,v,zc,inner_matrix,outer_matrix):
	point = np.array([[u,v,1.]]).T #(u,v,1)格式
	return zc*outer_matrix.dot(inner_matrix).dot(point)

if __name__ == '__main__':
	f = 6.5*1e-3 #焦距，单位m
	dx = 18.5*1e-6
	dy = 18.5*1e-6
	u0 = 173
	v0 = 130
	outer_matrix = np.array([[ 7.93143690e-01,3.22680771e-01,1.16408765e-01],
 [ 4.39507775e-02,-5.61276972e-01,2.97514737e-01],
 [-6.49131835e-07,-1.78813934e-06,9.99999642e-01],
 [ 7.73308635e-01,1.19861197e+00,7.59422362e-01]])
	inner_matrix = np.linalg.inv(np.array(
		[[f/dx,0,u0],[0,f/dy,v0],[0,0,1]]
		))
	points = np.array([
	[239,27,0.5,1.1,3.0], #x,y,xw,yw,zc
	[53,133,-0.5,0.7,2.9],
	[66,79,-0.3,0.8,2.4],
	[111,66,-0.2,0.83,2.1],
	[133,66,-0.1,0.86,1.9],
	[186,66,0.1,0.63,1.3],
	[266,111,0.2,0.5,1.0],
	[306,124,0.3,0.2,0.7],
	[348,199,0.4,0.1,0.5],
	[199,22,0.1,1.0,0.7],
	[288,199,0.3,0.4,0.3],
	[106,31,-0.2,1.0,1.8],
	])
	a = time.time()
	for it in range(1000):
		for point in points:
			print(D2toD3(point[0],point[1],point[-1],inner_matrix,outer_matrix)) #像素x像素y真实z内外
		exit()
	print(1000*(time.time()-a)/1000)