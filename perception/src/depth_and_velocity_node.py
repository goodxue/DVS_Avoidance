#!/usr/bin/env python
from allinc import *
from DeepCompute import deepEstMono
from CoordinateTranspose import D2toD3
from math import *
from KalmenFilter import Kalman
import rospy
from dvs_avoidance.msg import point2darray, point2i, pointarray
from geometry_msgs.msg import Vector3
import cv2
import numpy as np

gboxes = None


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


def Deep_Transform_Output(world_loc_last,each):

	#估计深度
	z_hat = deepEstMono(6.5*1e-3,20*1e-2,each[0]*18.5*1e-6)

	#估计这一帧的世界坐标
	world_loc_this = D2toD3(each[2]+0.5*each[0],each[3]+0.5*each[1],z_hat,inner_matrix,outer_matrix).T[0][:3] #坐标变换好像也没啥问题
		
	#粗暴地将上一帧的世界坐标与这一帧世界坐标相减除以时间来计算速度，暂时不加卡尔曼
	velocity = -(world_loc_this - world_loc_last)/1e-2
	return velocity,world_loc_this

def cluster_callback( cluster_point ):
	num_object = len(cluster_point.point2darray)
	if num_object == 0:
		return None
	if num_object >=2:
		print("number of obj is more than 1, not implemented!")
		#raise NotImplementedError

	global gboxes
	num_point = len(cluster_point.point2darray[0].pointarray)
	cluster_np = []
	for point_list in cluster_point.point2darray:
		point_np = []
		for point in point_list.pointarray:
			point_np.append(np.array([[point.x,point.y]]))
		cluster_np.append(np.array(point_np))
	#cluster_np = np.array(cluster_point.point2darray)
	# print("cluster-type:", type(cluster_point))
	# print("cluster_point.point2darray-type:", type(cluster_point.point2darray))
	# print("cluster_np-type:", type(cluster_np))
	# print("cluster_np[0]-type:", type(cluster_np[0]))
	gboxes = [cv2.boundingRect(cluster) for cluster in cluster_np]



if __name__ == '__main__':

	rospy.init_node('depth_and_velocity_node',anonymous=True)
	cluster_sub = rospy.Subscriber('/cluster_point',point2darray,cluster_callback)
	vel_pub = rospy.Publisher('/obj_velocity',Vector3,queue_size=1)
	loc_pub = rospy.Publisher('obj_location',Vector3,queue_size=1)
	rate = rospy.Rate(30)
	#获得相机内参、外参矩阵，根据拟合的数据
	inner_matrix,outer_matrix = computeMatrix() 

	'''
	读取数据并分割。
	程序输入数据格式：[宽、高、矩形左上角坐标x、矩形左上角坐标y]
	可以根据需要修改这里。
	'''

	# with open("1.txt","r") as f: 
	# 	data_list = f.read().split("\n")

	#数据格式的转换
	# data_list = [list(map(int,i.split(" "))) for i in data_list]

	#设定世界坐标初值，根据实验条件可以改变
	world_loc_last = np.array([0,1,3])

	#对每个矩形，进行迭代
	# for each in data_list:
	# 	velocity,world_loc_last=Deep_Transform_Output(world_loc_last,each)


		# '''
		# 此时算法可以输出。
		# 其输出格式为障碍物的速度[vx,vy,vz]，以及这一时刻的世界坐标
		# '''
	#rospy.loginfo("test?")
	while not rospy.is_shutdown():
		if not gboxes:
			continue
		#warning：假定gboxes内仅含有一个聚类结果！！
		for box in gboxes:
			velocity, world_loc_last=Deep_Transform_Output(world_loc_last,box)
		vel_obj = Vector3()
		vel_obj.x = velocity[0]
		vel_obj.y = velocity[1]
		vel_obj.z = velocity[2]
		loc_obj = Vector3()
		loc_obj.x = world_loc_last[0]
		loc_obj.y = world_loc_last[1]
		loc_obj.z = world_loc_last[2]
		vel_pub.publish(vel_obj)
		loc_pub.publish(loc_obj)
		rate.sleep()
			
