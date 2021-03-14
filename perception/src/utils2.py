import numpy as np 
import cv2
import pandas as pd
import random
import datetime
import sys
import copy
import time
#from dv import AedatFile
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
def cvshow(img,time=0,locx=500,locy=400):
	cv2.imshow("test",img)
	cv2.moveWindow("test", locx, locy)
	cv2.waitKey(time)


def readCsv(usecolname,path,sep="	",ifreverse=False,iftranspose=True,iflist=False):
	'''readCsv(使用列的名称：列表,文件路径,分隔符="	",是否要反转数据=False)：读取csv文件'''
	df = pd.read_csv(path,sep=sep)
	try:
		series = df[usecolname]
	except KeyError:
		raise KeyError("请检查您所用的列或分隔符是否正确")
	if iftranspose:
		series = np.array(series).T
	else:
		series = np.array(series)
	value_list = []
	if not iflist:
		for i in range(len(series)):
			if ifreverse:
				value_list.append(np.array(series[i][::-1]))
			else:
				value_list.append(np.array(series[i]))
	else:
		for i in range(len(series)):
			if ifreverse:
				value_list.extend(list(series[i][::-1]))
			else:
				value_list.extend(list(series[i]))
	return tuple(value_list)

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
	canvas.fill(0)
	for each in eventlist:
		cv2.circle(canvas,(int(each[0]),int(each[1])),1,(255,255,255),1)

	return eventlist,canvas

def createTimeWindows(eventlist,timestampstart,timestampend,size):
	if len(list(eventlist[0])) >= 5:
		timestamplist = list(set([x for x in eventlist[2] if timestampstart <= x <= timestampend]))
		eventlist = np.array(eventlist).T
		eventlist = [x for x in eventlist if x[2] in timestamplist]
	else:
		eventlist = np.array(eventlist).T
		timestamplist = list(set([x for x in eventlist[2] if timestampstart <= x <= timestampend]))
		eventlist = np.array(eventlist).T
		eventlist = [x for x in eventlist if x[2] in timestamplist]
	canvas = np.zeros(size,dtype=np.uint8)
	canvas.fill(0)
	for each in eventlist:
		cv2.circle(canvas,(int(each[0]),int(each[1])),1,(255,255,255),1)

	return eventlist,canvas

def dbscan(P,eps=3,minpts=2):
	pointlist = []
	for y in range(P.shape[1]):
		for x in range(P.shape[0]):
			if P[x,y] > 0:
				pointlist.append([x,y])
	pointlist = np.array(pointlist)
	db = skc.DBSCAN(eps, minpts).fit(pointlist)
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	cluster_list = []
	for i in range(n_clusters_):
		one_cluster = pointlist[labels == i]
		cluster_list.append([len(one_cluster),one_cluster])
	cluster_list.sort(key = lambda x:x[0],reverse = True)
	return cluster_list

def createCanvas(size,full=0):
	canvas = np.zeros(size,dtype=np.uint8)
	canvas.fill(full)
	return canvas

def diff(lastframe,thisframe,mode=1):
	if mode==0:
		thisframe = np.array(thisframe,dtype=np.float32)
		lastframe = np.array(lastframe,dtype=np.float32)
		subtract = thisframe - lastframe
		return subtract
	else:
		picture = cv2.absdiff(thisframe,lastframe)
		return picture
"""
def eventReader(inputfile,outputfile,iftranspose=False,ifwrite=False):
    with AedatFile(inputfile) as f:
        # events will be a named numpy array
        events = np.hstack([packet for packet in f['events'].numpy()])
        # Slice events
        with open(outputfile,"w+") as f:
            f.write("t x y p\n")
            for eachevent in events:
                f.write(str(eachevent[0])+" "+str(eachevent[1])+" "+str(eachevent[2])+" "+str(-1 if eachevent[3]==0 else 1)+"\n")
        eventlist = readCsv(["x","y","t","p"],outputfile," ",iftranspose=iftranspose)
        if not ifwrite:
            with open(outputfile,"w") as f:
            	f.write("")
        return eventlist
"""
def helper():
	print("readCsv(使用列的名称：列表,文件路径,分隔符="	",是否要反转数据=False,是否要转置=True,是否要转换成列表=False)：读取csv文件")
	print('event2Picture(eventtext = "gun_bullet_gnome.txt", size = (480,640,3))将事件可视化显示出来')
	print('video2Event(videoname = "QQ20200509142514.mp4", writefor = "eventsUAV.txt")将RGB视频转换为事件')
	print("createATime(eventlist,timestamp,size)截取一张包含timestamp的事件图，返回对应事件列表及图像")
	print("createTimeWindows(eventlist,timestampstart,timestampend,size)创建一张在timestamprange中的时间窗口")
	print("dbscan(P,eps=3,minpts=2)对P图像，半径为eps，最少点为2进行密度聚类，返回cluster_list")
	print("createCanvas(size,full)创建一个size，填充为full的画布")
	print("diff(lastframe,thisframe,mode=1)计算两张图片的差，mode=0表示允许负数，mode=1表示不允许")
	print("eventlist = eventReader(inputfile,outputfile,iftranspose=False,ifwrite=False)：将事件从admat格式转换为eventlist并且保存事件到记事本")