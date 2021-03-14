from allinc import *
from frontDeal import createTimeWin,generateCountImage
from models.shuffleNetV2Config import FasterNet

if __name__ == '__main__':
	torch.set_default_tensor_type(torch.FloatTensor)
	datadir = "DatasetTXT/"
	list_data = os.listdir(datadir)
	#Construct NN Model
	model = FasterNet([2,4,8,8,16,32,64,128],nr=3) 
	loss_fn = nn.MSELoss()
	y = torch.tensor([[0,200,-300]]) #vx=0,vy=200,vz=-300
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	for it, file in enumerate(list_data):
		#read events
		eventlist = readCsv(["x","y","t","p"],datadir+file," ",True,True,False)
		mintime,maxtime = np.min(eventlist[2]),np.max(eventlist[2])
		slice_time = 100
		deltatime = (maxtime-mintime)/slice_time
		eventlist = np.array(eventlist).T
		#update first frame
		canvas_this = np.zeros((350,350))
		eventlist2 = createTimeWin(eventlist,int(mintime),int(mintime+deltatime))
		canvas_this = generateCountImage(canvas_this,eventlist2)
		for t in range(1,slice_time):
			a = time.time()
			canvas_next = np.zeros((350,350))
			start = int(mintime + t*deltatime)
			end = int(mintime + (t+1)*deltatime)
			eventlist2 = createTimeWin(eventlist,start,end) #0.66ms
			canvas_next = generateCountImage(canvas_next,eventlist2) #2.6ms
			#Put 2 frames to the model
			x = torch.tensor([[canvas_this,canvas_next]],dtype=torch.float32) #18-22ms
			yhat = model(x) #7-10ms
			#print(1000*(time.time()-a)) #calculate the time
			#exit()
			loss = loss_fn(y,yhat)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print(it,loss.item())
			#update the following frames
			canvas_this = canvas_next

		save(model,"test1ep.pth")
