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
'''
def eventCounter(eventlist,canvas):
    timepic = np.zeros_like(canvas,dtype=np.uint8)
    for eachevent in eventlist:
        timepic[int(eachevent[1])][int(eachevent[0])] += 1
    return timepic

@njit
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
'''
parser = argparse.ArgumentParser()
parser.add_argument('--kernal_size',
                        type=int,
                        help="Training batch size.",
                        default=5)
parser.add_argument('--eps',
                        type=float,
                        help="Training batch size.",
                        default=3)
parser.add_argument('--minpts',
                        type=int,
                        help="Training batch size.",
                        default=2)

args = parser.parse_args()

#helper()
inputdata= "dvSave-2021_01_07_12_46_32.aedat4"
eventdata = eventReader(inputdata,"testobject.txt",ifwrite=True) #元组[x,y,t,p]
omegadata = omega_get(inputdata)
print(len(eventdata))
bbox1 = 0
timelen = 10000
f = open("result.txt","w") #
count = int((eventdata[-1][2]-eventdata[0][2])/timelen)
print(count)
v_total = []
pos_total = []
pre_img = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc,20.0,(346,260)) #width*height
eventc = 0 #计数
for i in range(count):
    #创建时间滑动窗口
    
    print("还差%d次"%(count-i))
    eventlist, canvas, eventc = createTimeWindows(eventdata,eventdata[0][2]+timelen*i,eventdata[0][2]+timelen*(i+1),(260,346),eventc) #数字串超过文件里面的数值也可以用？
    #cvshow(canvas)

    omegalen=int(len(omegadata)/count)
    omegalist = omegadata[i*omegalen:(i+1)*omegalen,:]
    eventlist = np.array(eventlist)
    eventlist = warp(eventlist[:,:3],omegalist,260,346)
    #创建countimage及归一化
    timepic = eventCounter(eventlist,canvas) #计数后的时间戳图像

    timepic = dealTimePicture(timepic) #归一化
    print(np.sum(timepic))
    
    #cvshow(timepic*255) #怎么区分是0-1图像还是0-255图像？

    #使用腐蚀操作
    kernal = np.ones((args.kernal_size,args.kernal_size),np.uint8)      #调参1
    timepic = cv2.erode(timepic,kernal,iterations=1) #腐蚀
    print(np.sum(timepic))
    #cvshow(timepic*255)
    image = np.ascontiguousarray(timepic)
    image = np.uint8(timepic)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image * 255
    image = image.astype(np.uint8)
    out.write(image)
    """
    if type(pre_img) == type(0):
        pre_img = image
    elif int(np.sum(timepic)) > 0 :
        pass
        #optical_flow(pre_img,image)

    if int(np.sum(timepic)) > 0 :
        index_max = np.where(timepic == np.max(timepic))
        y_max = int(np.max(index_max[0]))
        y_min = int(np.min(index_max[0]))
        x_max = int(np.max(index_max[1]))
        x_min = int(np.min(index_max[1]))
        topleft = (x_min, y_min)
        topright = (x_max,y_min)
        leftbottom = (x_min,y_max)
        rightbottom = (x_max, y_max)
        cv2.line(image,topleft,topright,(0,0,255))
        cv2.line(image,topleft,leftbottom,(0,0,255))
        cv2.line(image,leftbottom,rightbottom,(0,0,255))
        cv2.line(image,rightbottom,topright,(0,0,255))
        #image = cv2.rectangle(image,topleft,rightbottom)
        cvshow(image)
        pos = [(x_min+x_max)/2, (y_max+y_min)/2]
        pos_total.append(pos)
    """

    #密度聚类创建拟合矩形

    cluster_list = dbscan(timepic,eps = args.eps,minpts = args.minpts)
    cluster_list = [x[1] for x in cluster_list if x[0] >= 5]
    boundingbox = []
    center = []
    for each in cluster_list:
        xyright = np.array(each).T
        boundingbox.append([[np.min(xyright[0]),np.min(xyright[1])],[np.max(xyright[0]),np.max(xyright[1])]])
    
    #尝试利用事件的平移不变性观测聚类面积【以信号源强度作为指标】、光流方向【以矩形中心移动的位置作为指标】
    if not boundingbox:
        f.write("无物体偷袭\n")
        continue
    x = 0
    y = 0
    if bbox1 == 0:
        bbox1 = boundingbox
        for bound in boundingbox:
            x += (bound[0][0] + bound[1][0])
            y += (bound[0][1] + bound[1][1])
        
        bbox1 = (x/(2*len(boundingbox)), y/(2*len(boundingbox)), np.sum(timepic)/10)
    else:
        for bound in boundingbox:
            x += (bound[0][0] + bound[1][0])
            y += (bound[0][1] + bound[1][1])
        
        bbox2 = (x/(2*len(boundingbox)), y/(2*len(boundingbox)), np.sum(timepic)/10)
        velocity = [(bbox2[0] - bbox1[0]),(bbox2[1]-bbox1[1]),(bbox2[2]-bbox1[2])]
        v_total.append(velocity)
        escapex = random.randint(-100,100)/100
        escapey = random.randint(-100,100)/100
        escapez = -(velocity[0]*escapex+velocity[1]*escapey)/velocity[2]
        escapez = np.around(escapez,2)
        escape_velocity = [escapex,escapey,escapez]
        f.write("物体的速度"+str(velocity)+"无人机的逃脱速度"+str(escape_velocity)+'\n')
        #print("物体的速度",velocity,"无人机的逃脱速度",escape_velocity)
        bbox1 = bbox2
f.close	()
out.release()
cv2.destroyAllWindows()
plt.subplot(131)
plt.plot([x[0] for x in pos_total])
plt.legend(['pos_x'])
plt.subplot(132)
plt.plot([x[1] for x in pos_total])
plt.legend(['pos_y'])
plt.subplot(133)
plt.plot([x[2] for x in v_total])
plt.legend(['z'])
plt.title('kernel_siez:%s *%s eps:%s minpts:%s'%(args.kernal_size,args.kernal_size,args.eps,args.minpts))
plt.savefig('./pic3/xy_%s_%s_%s_%s.png'%(args.kernal_size,args.kernal_size,args.eps,args.minpts))

#plt.show()
