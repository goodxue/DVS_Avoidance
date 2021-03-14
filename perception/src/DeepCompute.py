from allinc import *

def deepEstStero(b,w,imgL,imgR):
	#双目深度估计
	stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	disp = stereo.compute(imgL,imgR)/1.0
	disp[disp==0] = np.inf
	return np.mean(abs(b*w/disp)) #返回物体的估计宽度

def deepEstMono(f,wreal, w): #尽量不要调函数，会慢
	return f*wreal/w

if __name__ == '__main__':
	#imgL = cv2.imread('2.jpg',0)
	#imgR = cv2.imread('1.jpg',0)
	#imgL = cv2.resize(imgL,(0,0),fx=0.2,fy=0.2)
	#imgR = cv2.resize(imgR,(0,0),fx=0.2,fy=0.2)
	print(deepEstMono(7.5*1e-3,20*1e-2,4.75*1e-4))