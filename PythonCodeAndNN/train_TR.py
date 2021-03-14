from allinc import *
from DeepCompute import deepEstMono

class SoftPlusModel(nn.Module):
	def __init__(self):
		super(SoftPlusModel,self).__init__()
		#模型D
		self.Layer1 = nn.ModuleList([
		nn.Linear(1,32),
		nn.Linear(32,128),
		nn.Linear(128,512),
		nn.Linear(512,512),
		nn.Linear(512,512),
		nn.Linear(512,512),
		nn.Linear(512,512),
		nn.Linear(512,512),]
		)
		self.Layer2 = nn.Linear(512,12)
	def forward(self,x):
		for layer in self.Layer1:
			x = layer(x)
			x = F.softplus(x)
		return self.Layer2(x)

if __name__ == '__main__':
	f = 6.5*1e-3 #焦距，单位m
	dx = 18.5*1e-6
	dy = 18.5*1e-6
	u0 = 173
	v0 = 130
	points = np.array([
	[239,27,0.5,1.1,3.0], #x,y,xw,yw,zc
	[53,133,-0.5,0.7,2.9],
	[66,79,-0.3,0.8,2.4],
	[111,66,-0.2,0.83,2.1],
	[133,66,-0.1,0.86,1.9],
	[186,66,0.1,0.63,1.3],
	[266,111,0.2,0.5,1.0],
	[306,124,0.3,0.2,0.7],
	[348,199,0.4,0.1,0.5],
	[199,22,0.1,1.0,0.7],
	[288,199,0.3,0.4,0.3],
	[106,31,-0.2,1.0,1.8],
	]).T
	uv = points[:2]
	zc = points[-1]
	fl = np.ones_like(zc)*f
	zreal = np.ones_like(zc)
	xwyw = points[2:4]
	train_x = np.vstack((uv,zreal,fl,zc)).T
	train_y = np.vstack((xwyw,zc,zreal)).T
	models = nn.Sequential(
		nn.Linear(1,32),
		nn.Sigmoid(),
		nn.Linear(32,128),
		nn.Sigmoid(),
		nn.Linear(128,128),
		nn.Sigmoid(),
		nn.Linear(128,128),
		nn.Sigmoid(),
		nn.Linear(128,128),
		nn.Sigmoid(),
		nn.Linear(128,128),
		nn.Sigmoid(),
		nn.Linear(128,12),
		).cuda()
	model = SoftPlusModel().cuda()
	model = load(model,"outermatrix_inv.pth")
	train_x=torch.tensor(train_x,dtype=torch.float32).cuda()
	train_y=torch.tensor(train_y,dtype=torch.float32).cuda()
	inner_matrix = torch.tensor(
		np.linalg.inv(np.array([[f/dx,0,u0],[0,f/dy,v0],[0,0,1]]))
		,dtype=torch.float32).cuda()
	loss_fn = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
	real_input = torch.tensor([[1.]]).cuda()
	for it in range(2500):
		totalloss = 0.
		for each in range(12):
			x_input = train_x[each]
			y_output = train_y[each]
			yhat = model(real_input) #生成tensor不随x变化
			yhat = yhat.reshape((4,3)) #转换为4行3列矩阵（直接求逆矩阵）
			#left = (x_input[-1]*x_input[:3]).unsqueeze(dim=1) #左侧：zc*(u,v,1)
			#right = inner_matrix.mm(torch.mm(yhat,y_output.unsqueeze(dim=1))) #右侧
			left = x_input[-1]*torch.mm(yhat,inner_matrix).mm(x_input[:3].unsqueeze(dim=1))
			right = y_output.unsqueeze(dim=1)
			loss = loss_fn(left,right)
			totalloss += loss
		print(yhat.detach().cpu().numpy())
		exit()
		print(it,totalloss.item())
		optimizer.zero_grad()
		totalloss.backward()
		optimizer.step()
	save(model,"outermatrix_inv.pth")