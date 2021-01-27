from allinc import *

class BasicBlock(nn.Module):
    def __init__(self,inc,outc):
        super(BasicBlock,self).__init__()
        self.block = nn.Sequential(
        nn.Conv2d(inc, outc, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(outc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(outc, outc, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=outc, bias=False),
        nn.BatchNorm2d(outc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(outc, outc, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(outc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
        )
    def forward(self,x):
    	return self.block(x)

class FasterNet(nn.Module):
    def __init__(self,scale = [3,6,12,24,48,96],nr = 10):
        super(FasterNet,self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(scale[0], scale[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(scale[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        self.body = nn.ModuleList([])

        for index in range(1,len(scale)-1):
        	self.body.append(BasicBlock(scale[index],scale[index+1]))

        self.viewing = torch.nn.AdaptiveMaxPool2d((1,1))
        self.regress = nn.Sequential(
        	nn.Linear(scale[-1],nr)
        	)

    def forward(self,x):
        x = self.conv1(x) #torch.Size([16, 6, 60, 80])
        for layer in self.body:
            x = layer(x)
        x = self.viewing(x).squeeze(dim=-1).squeeze(dim=-1)
        x = self.regress(x)
        return x


if __name__ == '__main__':
	model = FasterNet([3,6,6,12,12,24,24,48,48,96,92])
	x = torch.randn((16,3,240,320))
	y = model(x)
	print(y.shape)
	exit()
	save(model,"test.pth")