from allinc import *
from DeepCompute import deepEstMono
from CoordinateTranspose import D2toD3
from math import *
from KalmenFilter import Kalman


def angle_of_vector(v1, v2):
	#该函数的功能是计算向量之间的夹角
    pi = np.pi
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180

def computeMatrix():
	#该函数主要规定了超参数
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
	return inner_matrix,outer_matrix


if __name__ == '__main__':
	
	#获得相机内参、外参矩阵，根据拟合的数据
	inner_matrix,outer_matrix = computeMatrix() 

	'''
	读取数据并分割。
	程序输入数据格式：[宽、高、矩形左上角坐标x、矩形左上角坐标y]
	可以根据需要修改这里。
	'''

	with open("1.txt","r") as f: 
		data_list = f.read().split("\n")

	#数据格式的转换
	data_list = [list(map(int,i.split(" "))) for i in data_list]

	#如果需要可视化，那么就增加画布信息
	canvas = createCanvas((346,346),0)
	canvas2 = canvas.copy()

	#设定世界坐标初值，根据实验条件可以改变
	world_loc_last = np.array([0,1,3])

	#创建两个列表，分别为速度列表和速度均值列表，用于后续分析速度和均值
	v_list = []
	mean_list = []

	#对每个矩形，进行迭代
	for each in data_list:

		#绘制矩形，如果需要可视化
		#canvas = cv2.rectangle(canvas, (each[2],each[3]), (each[2]+each[0],each[3]+each[1]), (255,255,255), 3)
		
		#估计深度
		z_hat = deepEstMono(6.5*1e-3,20*1e-2,each[0]*18.5*1e-6)

		#估计这一帧的世界坐标
		world_loc_this = D2toD3(each[2]+0.5*each[0],each[3]+0.5*each[1],z_hat,inner_matrix,outer_matrix).T[0][:3] #坐标变换好像也没啥问题
		
		#粗暴地将上一帧的世界坐标与这一帧世界坐标相减除以时间来计算速度，暂时不加卡尔曼
		velocity = -(world_loc_this - world_loc_last)/1e-2

		#判断是否为第一次监视到物体的存在
		if not v_list or len(v_list)<=2:
			#第一次监视物体存在，如果采用均值滤波，则需要将速度设置为均值
			#mean = velocity

			#将速度添加到速度列表中
			v_list.append(velocity.tolist())
		else:

			#先将速度添加到速度列表中
			v_list.append(velocity.tolist())

			#如果需要，则计算均值
			mean = np.sum(np.array(v_list[2:]),axis=0)/len(v_list[2:])

			#最后将均值添加到均值列表中
			mean_list.append(mean)


		'''
		#如果后期需要增加卡尔曼滤波，则可以修改这行注释
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

		#更新世界坐标
		world_loc_last = world_loc_this

		'''
		此时算法可以输出。
		其输出格式为障碍物的速度[vx,vy,vz]，调用v_list[-1]即可
		如果觉得v_list多余，也可以删除。
		这里添加v_list的目的是方便画图
		'''



	#作图代码，主要做关于某一参考直线的速度方向，方向为常数则说明计算稳定。画图过程可忽略

	#如果使用均值滤波，则将速度列表更新为均值列表
	v_list = mean_list

	v_list =v_list[2:-3]
	it = [i for i in range(len(v_list))]
	v_list = np.array(v_list)
	cosine_escape = []
	for v_list1 in v_list:
		cosine_ = angle_of_vector((v_list1[0],v_list1[1],v_list1[2]),(1,0,0))
		cosine_escape.append(cosine_)
	plt.plot(it,np.array(cosine_escape),label="Escape Cosine")
	plt.legend()
	plt.show()
