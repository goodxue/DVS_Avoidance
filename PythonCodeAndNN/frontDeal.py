from allinc import *
'''
	listfile = os.listdir("DVSDataset")
	
	for each in listfile:
		eventReader("DVSDataset/"+each,outputdir+each.rstrip(".aedat4")+".txt",iftranspose=False,ifwrite=True)
		'''

def createTimeWin(eventlist,begin,end):
	#输入的事件应该是N*4的(x,y,t,p)
	eventlist = eventlist.T
	arrbool = np.logical_and(eventlist[2]>=begin,eventlist[2]<=end)
	eventlist = eventlist.T[arrbool]
	return eventlist

def generateCountImage(canvas,eventlist,norm=False,trans =1):
	for point in eventlist:
		#listi[int(point[0])][int(point[1])].append(point[2]/100000000)
		canvas[int(point[0]),int(point[1])] += 255
	if norm:
		canvas = (canvas-np.min(canvas))/(np.max(canvas)-np.min(canvas))
	if trans:
		return canvas.T
	return canvas

def constructTimeImage(listi,canvas):
	listi = np.array(listi)
	sumT = np.zeros_like(listi)
	for x in range(listi.shape[0]):
		for y in range(listi.shape[1]):
			sumT[x,y] = np.sum(listi[x][y])
	return sumT/canvas,np.mean(sumT)


if __name__ == '__main__':
	datadir = "DatasetTXT/"
	eventlist = readCsv(["x","y","t","p"],datadir+"dvSave-2021_01_15_15_39_21.txt"," ",True,True,False)
	mintime,maxtime = np.min(eventlist[2]),np.max(eventlist[2])
	slice_time = 100
	deltatime = (maxtime-mintime)/slice_time
	eventlist = np.array(eventlist).T
	#a = time.time()
	for t in range(slice_time):
		canvas = np.zeros((350,350))
		start = int(mintime + t*deltatime)
		end = int(mintime + (t+1)*deltatime)
		eventlist2 = createTimeWin(eventlist,start,end)
		print(eventlist2.shape)
		canvas = generateCountImage(canvas,eventlist2)
		#Timage,meanT = constructTimeImage(listi,canvas)
		cvshow(canvas)
	#print(1000*(time.time()-a)/slice_time)