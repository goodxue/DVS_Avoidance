# coding=UTF-8
from utils2 import *
import cv2
import numpy as np
import copy
import time
#from numba import njit
#from egofunction import *
import matplotlib.pyplot as plt
import argparse
from PIL import Image

#初始化
parser = argparse.ArgumentParser()
parser.add_argument('--kernal_size',
                        type=int,
                        help="erode",
                        default=2)
parser.add_argument('--eps',
                        type=float,
                        help="DBSCAN param",
                        default=3)
parser.add_argument('--minpts',
                        type=int,
                        help="DBSCAN param",
                        default=2)
parser.add_argument('--inputpath',
                        type=str,
                        help="input path",
                        default="./dvSave-2021_01_07_12_46_32.aedat4")
parser.add_argument('--outputpath',
                        type=str,
                        help="output path",
                        default="./test1")
args = parser.parse_args()

inputflod = "./data"
inputdata= "dvSave-2021_01_07_12_46_32.aedat4"
eventdata = eventReader(args.inputpath,"testobject.txt",ifwrite=True) #元组[x,y,t,p]
omegadata = omega_get(args.inputpath)
bbox1 = 0
timelen = 10000
ff = open("result.txt","w") #
count = int((eventdata[-1][2]-eventdata[0][2])/timelen) #时间窗口个数
v_total = []
pos_total = []
pre_img = 0
eventc = 0 #计数
f = 5.5*1e-3 #焦距
wreal = 2*1e-2 #物体真实宽度
h = 260 #图像高
w = 346 #图像宽
u0 = 173
v0 = 130
eventdata = np.array(eventdata)
P = np.mat([[1, 0], [0, 1]])
X = np.zeros((2,3))
X = np.mat(X)
t1 = time.time()

for i in range(count):
    #创建时间滑动窗口    
    print("还差%d次"%(count-i))
    start = eventdata[0][2]+timelen*i
    end =  eventdata[0][2]+timelen*(i+1)
    eventlist = createTimeWin(eventdata,start,end)

    #自我运动补偿
    omegalen=int(len(omegadata)/count)
    omegalist = omegadata[i*omegalen:(i+1)*omegalen,:]
    eventlist = warp(eventlist[:,:3],omegalist,h,w)
    
    #生成count image 和timeimage
    canvas = np.zeros((h,w))
    countpic, timepic = generateCountImageAndTimeimage(canvas,eventlist)
    cvsave(args.outputpath+"/count_image"+str(i)+".jpg",countpic*255)
    cvsave(args.outputpath+"/time_image"+str(i)+".jpg",timepic*255)

    #生成mean_timeimage以及rho
    rho ,m_timepic= dealTimePicture(timepic,timelen) #归一化
    cvsave(args.outputpath+"/mean_time_image"+str(i)+".jpg",m_timepic*255)
    cvsave(args.outputpath+"/thresold_image"+str(i)+".jpg",rho*255)
    
    #使用腐蚀操作
    kernal = np.ones((args.kernal_size,args.kernal_size),np.uint8)      #调参1
    rho = cv2.erode(rho,kernal,iterations=1) #腐蚀
    cvsave(args.outputpath+"/final_image"+str(i)+".jpg",rho*255)
    #cvshow("count_image",countpic*255) #countimage

    #密度聚类(仿制)
    cluster = []
    for i in range(1,h-1):
        for j in range(1,w-1):
            if rho[i][j]==1 and (rho[i-1][j]==1 or rho[i+1][j]==1 or rho[i-1][j-1]==1 or rho[i][j-1]==1 or rho[i+1][j+1]==1 or rho[i-1][j+1]==1 or rho[i][j+1]==1 or rho[i+1][j+1]==1):
                cluster.append([i,j])

    #无聚类结果则不进行后续步骤
    if  not cluster:
        continue
    
    #聚类效果显示
    #temp = np.zeros((h,w))
    #for t in cluster:
    #    temp[t[0]][t[1]] = 255
    #cvshow("cluster",temp)
    #print("stop")

    #提取物体外框矩形角点x,y以及中心点
    cluster = np.array(cluster)
    x_min = np.min(cluster[:,1])
    x_max = np.max(cluster[:,1])
    y_min = np.min(cluster[:,0])
    y_max = np.max(cluster[:,0])
    center_x = (x_min+x_max)/2
    center_y = (y_min+y_max)/2

    #深度估计(提取z)
    width = x_max - x_min
    height = y_max - y_min
    if width != 0 or height != 0:
        depth = f*wreal/(width if width!= 0 else height)
    else:
        depth = 0
        print("深度错误")
        continue

    #坐标转换
    #拟合出来的相机矩阵
    outer_matrix = np.array([[ 7.93143690e-01,3.22680771e-01,1.16408765e-01],
    [ 4.39507775e-02,-5.61276972e-01,2.97514737e-01],
    [-6.49131835e-07,-1.78813934e-06,9.99999642e-01],
    [ 7.73308635e-01,1.19861197e+00,7.59422362e-01]])
    xyz = D2toD3(outer_matrix,center_x,center_y,depth,f)
    xyz = np.array(xyz[:3]).T

    #卡尔曼滤波
    noise = np.round(np.random.normal(0, 1, 3), 2)
    Z = np.mat(xyz) + np.mat(noise)# 将z的观测值和噪声相加
    X,P = Kalman(Z,X,P)
    pos_total.append(X[0,:])
    v_total.append(X[1,:])


ff.write(args.inputpath+":平均时长"+str((time.time()-t1)/count)+'s \n')
ff.close	()
#out.release()
#cv2.destroyAllWindows()

#画图
pos_total = np.array(pos_total)
v_total = np.array(v_total)
plt.subplot(231)
plt.plot(pos_total[:,:,0])
plt.legend(['pos_x'])
plt.subplot(232)
plt.plot(pos_total[:,:,1])
plt.legend(['pos_y'])
plt.subplot(233)
plt.plot(pos_total[:,:,2])
plt.legend(['pos_z'])
plt.subplot(234)
plt.plot(v_total[:,:,0])
plt.legend(['v_x'])
plt.subplot(235)
plt.plot(v_total[:,:,1])
plt.legend(['v_y'])
plt.subplot(236)
plt.plot(v_total[:,:,2])
plt.legend(['v_z'])
plt.title('kernel_siez:%s *%s eps:%s minpts:%s'%(args.kernal_size,args.kernal_size,args.eps,args.minpts))
#plt.savefig('./pic3/xy_%s_%s_%s_%s.png'%(args.kernal_size,args.kernal_size,args.eps,args.minpts))

plt.show()
