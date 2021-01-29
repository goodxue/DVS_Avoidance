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

def cvshow(img,time=0):
    #if  type(img) is not np.uint8:
    #    img=(img*255).astype(np.uint8)
    #cv2.imwrite('test.png',img)
    cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    cv2.imshow('test',img)
    cv2.moveWindow("test", 100, 800) #？
    
    cv2.waitKey(time)
    cv2.destroyAllWindows() #新加
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

def event2Picture(eventtext = "gun_bullet_gnome.txt", size = (480,640,3)):
    eventlist = readCsv(["x","y","t","p"],eventtext," ",iftranspose=False)
    timestamp = list(set(readCsv(["t"],eventtext," ",iftranspose=False,iflist=True)))
    timestamp.sort()
    eventlist_copy = copy.deepcopy(eventlist)
    canvas = np.zeros(size,dtype=np.uint8)
    canvas.fill(0)
    white = canvas.copy()
    for eachtime in range(len(timestamp)):
        eventlist = [x for x in eventlist_copy if x[2] == timestamp[eachtime]]
        for each in eventlist:
            if each[3] == 1:
                cv2.circle(canvas,(int(each[0]),int(each[1])),1,(0,0,255),1)
            else:
                cv2.circle(canvas,(int(each[0]),int(each[1])),1,(255,0,0),1)
        cvshow(canvas,1)
        canvas = white.copy()

def video2Event(videoname = "QQ20200509142514.mp4", writefor = "eventsUAV.txt"):
    video = cv2.VideoCapture(videoname)
    #处理视频
    threshold = 40
    ret,lastframe = video.read()
    lastframe = cv2.cvtColor(lastframe,cv2.COLOR_BGR2GRAY)
    event_list = []
    while True:
        try:
            ret, thisframe = video.read()
            thisframe_color = thisframe.copy()
            thisframe = cv2.cvtColor(thisframe,cv2.COLOR_BGR2GRAY)
            thisframe = np.array(thisframe,dtype=np.float32)
            lastframe = np.array(lastframe,dtype=np.float32)
            subtract = thisframe - lastframe
            for y in range(subtract.shape[0]):
                for x in range(subtract.shape[1]):
                    if subtract[y][x] >= threshold:
                        #(x,y,t,p)红色表示p=1，蓝色表示p=-1
                        event_list.append((x,y,video.get(0)/1000,1))
                        thisframe_color[y][x][2] = 255
                        thisframe_color[y][x][1] = thisframe_color[y][x][0] = 0
                    elif subtract[y][x] <= -threshold:
                        event_list.append((x,y,video.get(0)/1000,-1))
                        thisframe_color[y][x][0] = 255
                        thisframe_color[y][x][1] = thisframe_color[y][x][2] = 0
                    else:
                        thisframe_color[y][x][0] = 0
                        thisframe_color[y][x][1] = thisframe_color[y][x][2] = 0		
            cv2.imshow("name",thisframe_color)
            lastframe = thisframe
            if cv2.waitKey(1) & 0xFF == 27: #27位esc退出键
                break
        except:
            video.release()
            cv2.destroyAllWindows()
            with open(writefor,"w+") as f:
                for event in event_list:
                    f.write(str(event[0])+","+str(event[1])+","+str(event[2])+","+str(event[3])+"\n")
            break

@timefun
def createATime(eventlist,timestamp,size):
    if len(list(eventlist[0])) >= 5:
        timestamplist = list(set([x for x in eventlist[2] if x == timestamp]))
        eventlist = np.array(eventlist).T
        eventlist = [x for x in eventlist if x[2] in timestamplist]
    else:
        eventlist = np.array(eventlist).T
        timestamplist = list(set([x for x in eventlist[2] if x == timestamp]))
        eventlist = np.array(eventlist).T
        eventlist = [x for x in eventlist if x[2] in timestamplist]
    canvas = np.zeros(size,dtype=np.uint8)
    canvas.fill(0) #没必要带参数吧
    for each in eventlist:
        cv2.circle(canvas,(int(each[0]),int(each[1])),1,(255,255,255),1)

    return eventlist,canvas

@timefun
def createTimeWindows(eventlist,timestampstart,timestampend,size,count):
    #如果行数大于5
    
    """
    if len(list(eventlist[0])) >= 5:
        timestamplist = list(set([x for x in eventlist[2] if timestampstart <= x <= timestampend])) #set?
        eventlist = np.array(eventlist).T #为什么要转置呢？
        eventlist = [x for x in eventlist if x[2] in timestamplist] #这部分好像没啥卵用
    else:
        eventlist = np.array(eventlist).T
        timestamplist = [x for x in eventlist[2] if (x>=timestampstart  and x<= timestampend)]
        eventlist = np.array(eventlist).T
        eventlist = [x for x in eventlist if x[2] in timestamplist]
    """
    
    eventlist_select = []
    eventlist = np.array(eventlist)
    for event in eventlist[count:][:]:
        if event[2]<=timestampend :
            eventlist_select.append(event)
            count = count + 1
        else:
            break
    eventlist = eventlist_select
    
    canvas = np.zeros(size,dtype=np.uint8)
    canvas.fill(0)
    for each in eventlist:
        cv2.circle(canvas,(int(each[0]),int(each[1])),1,(255,255,255),1) #画点

    return eventlist,canvas,count

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

@timefun
def createCanvas(size,full=0):
    canvas = np.zeros(size,dtype=np.uint8)
    canvas.fill(full)
    return canvas
#此函数无引用？
def diff(lastframe,thisframe,mode=1):
    if mode==0:
        thisframe = np.array(thisframe,dtype=np.float32)
        lastframe = np.array(lastframe,dtype=np.float32)
        subtract = thisframe - lastframe
        return subtract
    else:
        picture = cv2.absdiff(thisframe,lastframe)
        return picture

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


#输入:两个都是三通道rgb图像
#功能：计算光流并可视化
def optical_flow(frame1,frame2):
    frame1 = means_filter(frame1,3)
    frame2 = means_filter(frame2,3)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0) #光流计算，flow_size:[height,width,2]
    #可视化
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1]) #...表示前面所有维数
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rgb = means_filter(rgb,3)
    cvshow(rgb)
import numpy as np


def means_filter(input_image, filter_size):
    '''
    三通道图像均值滤波器：太慢了！
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像
    注：此实现滤波器大小必须为奇数且 >= 3
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本
    filter_template = np.ones((filter_size, filter_size))  # 空间滤波器模板
    pad_num = int((filter_size - 1) / 2)  # 输入图像需要填充的尺寸
    input_image_cp = np.pad(input_image_cp, ((pad_num,pad_num), (pad_num,pad_num), (0,0)), mode="constant", constant_values=0)  # 填充输入图像
    m, n,  c = input_image_cp.shape  # 获取填充后的输入图像的大小
    output_image = np.copy(input_image_cp)  # 输出图像
    # 空间滤波
    for k in range(c):
        for i in range(pad_num, m - pad_num):
            for j in range(pad_num, n - pad_num):
                output_image[i, j,k] = np.sum(filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1,k]) / (filter_size ** 2)
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num, :]  # 裁剪

    return output_image

#角速度列表获取(new)
def omega_get(input_file):
    gyros = []

    with AedatFile(input_file) as f :
        for e in f['imu']:
            gyros.append(e.gyroscope)  
    gyros =  np.array(gyros)
    return gyros

"""
python版:
功能:自运动补偿(new)
输入:
events:[x,y,t]n  data_type:numpy size:3*n
omegas:[w_x,w_y,w_z]m data_type:numpy size:3*m
height,width:图像高宽 int
输出：
event_news:[x,y,t]n  data_type:numpy
"""
def warp(events, omegas,height,width):
    omega_bar = omegas.mean(axis=0)   
    #omega_bar = omega_bar/(omegas.len()+1) #size:1*3
    #omega_bar = np.array(omega_bar)
    omega_bar = omega_bar*3.14/180
    R = [[1,-omega_bar[-1],omega_bar[1]],[omega_bar[-1],1,-omega_bar[0]],[0,0,1]]  #rodriguez公式简化得到的旋转矩阵 size:3*3
    R = np.array(R)
    #t0 = events[0][-1]
    #for event in events:)
    #	event[-1] -= t0
    #theta = omega_bar.T * events[:][3] # size:3*1 & 1*m   data:[theta_x,theta_y,theta_z]m
    events_new = np.matmul(events ,R)#3*3 x3*n 自补偿后的事件信息
    events_new =events_new.astype(int)
    event_news = []
    for event in events_new:
        if event[1]>=height or event[0]>=width or event[1]<0 or event[0]<0:
            continue
        else:
            event_news.append(event)
    event_news = np.array(event_news)
    return event_news

#egofunction
def eventCounter(eventlist,canvas):
	timepic = np.zeros_like(canvas,dtype=np.uint8)
	for eachevent in eventlist:
		timepic[int(eachevent[1])][int(eachevent[0])] += 1
	return timepic

def dealTimePicture(timepic):
	#归一化到0-1中
	maxval = np.max(timepic)
	minval = np.min(timepic)
	timepic = (timepic - minval)/(maxval-minval)
	#添加阈值：大于miu+2sigma的取出来
	miu = np.mean(timepic)
	sigma = np.std(timepic)
	for y in range(timepic.shape[0]):
		for x in range(timepic.shape[1]):
			if timepic[y][x] <= miu+2*sigma:
				timepic[y][x] = 0
			else:
				timepic[y][x] = 1	
	return timepic

def helper():
    print("readCsv(使用列的名称：列表,文件路径,分隔符="	",是否要反转数据=False,是否要转置=True,是否要转换成列表=False)：读取csv文件")
    print('event2Picture(eventtext = "gun_bullet_gnome.txt", size = (480,640,3))将事件可视化显示出来')
    print('video2Event(videoname = "QQ20200509142514.mp4", writefor = "eventsUAV.txt")将RGB视频转换为事件')  #为什么要做这个函数
    print("createATime(eventlist,timestamp,size)截取一张包含timestamp的事件图，返回对应事件列表及图像") #单独调用
    print("createTimeWindows(eventlist,timestampstart,timestampend,size)创建一张在timestamprange中的时间窗口")
    print("dbscan(P,eps=3,minpts=2)对P图像，半径为eps，最少点为2进行密度聚类，返回cluster_list")
    print("createCanvas(size,full)创建一个size，填充为full的画布")
    print("diff(lastframe,thisframe,mode=1)计算两张图片的差，mode=0表示允许负数，mode=1表示不允许")
    print("eventReader(inputfile,outputfile,iftranspose=False,ifwrite=False)：将事件从admat格式转换为eventlist并且保存事件到记事本")