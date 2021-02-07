# coding=UTF-8
import numpy as np 
from cv2 import cv2
import pandas as pd
import random
import datetime
import sys
import copy
import time
from dv import AedatFile
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from functools import wraps

#保存图像
def cvsave(name,img,time=0):
    cv2.imwrite(name,img)

#显示图像
def cvshow(name,img,time=0):    
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,img)
    cv2.moveWindow(name, 100, 800) 
    cv2.waitKey(time)
    cv2.destroyAllWindows()

#计算时间
def timefun(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result
    return measure_time

@timefun
def readCsv(usecolname,path,sep="	",ifreverse=False,iftranspose=True,iflist=False):
    '''readCsv(使用列的名称：列表,文件路径,分隔符="	",是否要反转数据=False)：读取csv文件'''
    df = pd.read_csv(path,sep=sep) #类似于读取excel表格形式的数据
    try:
        series = df[usecolname] #？？？转成矩阵？写入时是[t,x,y,p],读取时是[x,y,t,p]
    except KeyError:
        raise KeyError("请检查您所用的列或分隔符是否正确")
    if iftranspose:
        series = np.array(series).T  #转置
    else:
        series = np.array(series)  #numpy类型
    value_list = []
    if not iflist:
        #len(series)返回行数
        for i in range(len(series)):
            if ifreverse:
                value_list.append(np.array(series[i][::-1])) #反转数据，都已经是array类型了为什么还要再强制类型一次？
            else:
                value_list.append(np.array(series[i]))
    else:
        for i in range(len(series)):
            if ifreverse:
                value_list.extend(list(series[i][::-1]))  #extend:在已存在的列表中添加新的列表内容 
            else:
                value_list.extend(list(series[i]))
    return value_list

#角速度列表获取(new)
@timefun
def omega_get(input_file):
    gyros = []

    with AedatFile(input_file) as f :
        for e in f['imu']:
            gyros.append(e.gyroscope)  
    gyros =  np.array(gyros)
    return gyros

#离线读取事件
@timefun
def eventReader(inputfile,outputfile,iftranspose=False,ifwrite=False):
    with AedatFile(inputfile) as f:
        # events will be a named numpy array 官网api,将事件数据转换为numpy矩阵
        #np.hstack:沿着水平方向堆叠数组
        events = np.hstack([packet for packet in f['events'].numpy()]) #n*4
        print(len(events))
        # Slice events
        with open(outputfile,"w+") as f:
            f.write("t x y p\n")
            for eachevent in events:
                f.write(str(eachevent[0])+" "+str(eachevent[1])+" "+str(eachevent[2])+" "+str(-1 if eachevent[3]==0 else 1)+"\n")
        eventlist = readCsv(["x","y","t","p"],outputfile," ",iftranspose=iftranspose)  #元组，所以为什么不直接把events转成元组而是绕一个大弯？
        
        if not ifwrite:
            with open(outputfile,"w") as f:
                f.write("")
        return eventlist

#创建时间窗口
@timefun
def createTimeWin(eventlist,begin,end):
    #输入的事件应该是N*4的(x,y,t,p)
    eventlist = eventlist.T
    arrbool = np.logical_and(eventlist[2]>=begin,eventlist[2]<=end)
    eventlist = eventlist.T[arrbool]
    return eventlist

"""
功能:自运动补偿(new)
输入:
events:[x,y,t]n  data_type:numpy size:3*n
omegas:[w_x,w_y,w_z]m data_type:numpy size:3*m
tl:int
输出：
event_new:[x,y,t]n  data_type:numpy
"""
@timefun
def warp(events, omegas,height,width):
    omega_bar = omegas.mean(axis=0)   
    #omega_bar = omega_bar/(omegas.len()+1) #size:1*3
    #omega_bar = np.array(omega_bar)
    omega_bar = omega_bar*3.14/180
    R = [[1,-omega_bar[-1],omega_bar[1]],[omega_bar[-1],1,-omega_bar[0]],[0,0,1]]  #rodriguez公式简化得到的旋转矩阵 size:3*3
    R = np.array(R)
    t0 = events[0][-1]
    for event in events:
        event[-1] = event[-1] - t0
    #theta = omega_bar.T * events[:][3] # size:3*1 & 1*m   data:[theta_x,theta_y,theta_z]m
    events_new = np.matmul(events ,R)#3*3 x3*n 自补偿后的事件信息
    events_new =events_new.astype(int)#.T
    event_news = []
    for event in events_new:
        if event[1]>=height or event[0]>=width or event[1]<0 or event[0]<0:
            continue
        else:
            event_news.append(event)
    event_news = np.array(event_news)
    return event_news

#产生countimage 和timeimage
@timefun
def generateCountImageAndTimeimage(canvas,eventlist,norm=False):
    timeimage = canvas.copy()
    for point in eventlist:
        #listi[int(point[0])][int(point[1])].append(point[2]/100000000)
        canvas[int(point[1]),int(point[0])] += 1
        timeimage[int(point[1]),int(point[0])] += point[2]
    for h in range(canvas.shape[0]):
        for w in range(canvas.shape[1]):
            if canvas[h][w] == 0:
                continue
            else:
                timeimage[h][w] = timeimage[h][w] / canvas[h][w]
    if norm:
        canvas = (canvas-np.min(canvas))/(np.max(canvas)-np.min(canvas))
    return canvas, timeimage

#timeimage归一化
@timefun
def dealTimePicture(timepic,timelen,threshold=0.98):
    #归一化到0-1中
    maxval = np.max(timepic)
    minval = np.min(timepic)
    #m_timepic = np.zeros_like(timepic)
    miu = np.mean(timepic)
    timepic = (timepic - miu)/(maxval-minval)
    m_timepic = timepic.copy()
    #添加阈值：大于miu+2sigma的取出来
    miu = np.mean(timepic)
    sigma = np.std(timepic)
    threshold = miu+2*sigma
    for y in range(timepic.shape[0]):
        for x in range(timepic.shape[1]):
            if timepic[y][x] <= 0.9 : #之后变量化
                timepic[y][x] = 0
            else:
                timepic[y][x] = 1	
    return timepic, m_timepic

#密度聚类
@timefun
def dbscan(P,eps=3,minpts=2):
    pointlist = []
    for y in range(P.shape[1]):
        for x in range(P.shape[0]):
            if P[x,y] > 0:
                pointlist.append([x,y]) #提供非0点的坐标
    if  not pointlist:
        return np.array([])
    pointlist = np.array(pointlist)
    db = skc.DBSCAN(eps, minpts).fit(pointlist) #核心算法DBSCAN密度聚类
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_list = []
    for i in range(n_clusters_):
        one_cluster = pointlist[labels == i]
        cluster_list.append([len(one_cluster),one_cluster])
    cluster_list.sort(key = lambda x:x[0],reverse = True)
    return cluster_list

#深度估计（单目）
@timefun
def deepEstMono(f,wreal, w): #尽量不要调函数，会慢
    return f*wreal/w

@timefun
def deepEstStero(b,w,imgL,imgR):
	#双目深度估计
	stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	disp = stereo.compute(imgL,imgR)/1.0
	disp[disp==0] = np.inf
	return np.mean(abs(b*w/disp)) #返回物体的估计宽度

#坐标转换
@timefun
def D2toD3(outer_matrix,u,v,zc,f,dx=18.5*1e-6,dy=18.5*1e-6):
    inner_matrix = np.linalg.inv(np.array(
        [[f/dx,0,173],[0,f/dy,130],[0,0,1]]
        ))
    point = np.array([[u,v,1.]]).T #(u,v,1)格式
    return zc*outer_matrix.dot(inner_matrix).dot(point)

#卡尔曼滤波
#Z:传感器数据+噪声(x,y,z)i
#P :2x2
#X:2x3
@timefun
def Kalman(Z,X,P):
    deltat = 1e-3 #1ms
    F = np.mat([[1, deltat], [0, 1]])
    Q = np.mat([[0.0001, 0], [0, 0.0001]])
    H = np.mat([1, 0])
    R = np.mat([1])

    x_predict = F * X  #2x2 *2x3 =2x3
    p_predict = F * P * F.T + Q #2x2*2x2 * 2x2 + 2x2=2x2
    K = p_predict * H.T / (H * p_predict * H.T + R) # 2x2*2x1 / (1x2 * 2x2 * 2x1 + 1) = 2x1

    X = x_predict + K *(Z - H * x_predict) #2x3 + 2x1*(1x3 - 1x2*2*3) = 2x3
    P = (np.eye(2) - K * H) * p_predict #(2x2 - 2x1*1x2)*2x2 = 2*2
    return X,P 


