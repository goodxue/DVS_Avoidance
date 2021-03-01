from allinc import *
from DeepCompute import deepEstMono
from CoordinateTranspose import D2toD3
from math import *
from KalmenFilter import Kalman


def angle_of_vector(v1, v2):
    pi = np.pi
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180

def computeMatrix():
	f = 7.5*1e-3 #焦距，单位m
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
	return inner_matrix,outer_matrix

if __name__ == '__main__':
	inner_matrix,outer_matrix = computeMatrix()
	with open("1.txt","r") as f:
		data_list = f.read().split("\n")
	data_list = [list(map(int,i.split(" "))) for i in data_list]
	canvas = createCanvas((346,346),0)
	canvas2 = canvas.copy()
	world_loc_last = np.array([0,1,3])
	v_list = []
	mean_list = []
	for each in data_list:
		#canvas = cv2.rectangle(canvas, (each[2],each[3]), (each[2]+each[0],each[3]+each[1]), (255,255,255), 3)
		z_hat = deepEstMono(6.5*1e-3,20*1e-2,each[0]*18.5*1e-6)#深度估计较为准确
		world_loc_this = D2toD3(each[2]+0.5*each[0],each[3]+0.5*each[1],z_hat,inner_matrix,outer_matrix).T[0][:3] #坐标变换好像也没啥问题
		velocity = -(world_loc_this - world_loc_last)/1e-2

		if not v_list or len(v_list)<=2:
			mean = velocity
			v_list.append(velocity.tolist())
		else:
			v_list.append(velocity.tolist())
			#exit()
			mean = np.sum(np.array(v_list[2:]),axis=0)/len(v_list[2:])
			mean_list.append(mean)
			#print(mean)
		#print(v_list)


		'''
		if not v_list:
			xmk_1 = np.array([0.2,0.8,2.27,0,0,0]).T
			xk_1 = np.hstack((world_loc_this,velocity)).T
			v_list.append(velocity)
		else:
			#进行卡尔曼滤波
			xmk_1 = Kalman(xmk_1,xk_1,3,3)
			xk_1 = np.hstack((world_loc_this,velocity))
			v_list.append(xmk_1[:3])
		'''
		#print(v_list[-1])
		#cvshow(canvas)
		#canvas = canvas2.copy()
		world_loc_last = world_loc_this

	#exit()
	#作图
	v_list = mean_list
	v_list =v_list[2:-3]
	it = [i for i in range(len(v_list))]
	v_list = np.array(v_list)
	cosine_escape = []
	for v_list1 in v_list:
		cosine_ = angle_of_vector((v_list1[0],v_list1[1],v_list1[2]),(1,0,0))
		cosine_escape.append(cosine_)
	#plt.plot(it,v_list[0],label="Vx")
	#plt.plot(it,v_list[1],label="Vy")
	#plt.plot(it,v_list[2],label="Vz")
	plt.plot(it,np.array(cosine_escape),label="Escape Cosine")
	plt.legend()
	plt.show()
