from allinc import *
from models.shuffleNetV2Config import FasterNet


torch.set_default_tensor_type(torch.FloatTensor)

#训练路径
path = "MNIST/train"
pathtest = "MNIST/val"
#设置batch_size为256——更改为16，由于内存不足
batch_size = 16
#类别数
num_of_class = 10

#获取图片标签，输入图片的路径获取
def get_tag(img):
	tagarray = readCsv(["picname","number"],"MNIST/train.txt"," ")
	tagindex = list(tagarray[0]).index(img)
	count = tagarray[1][tagindex]
	return count

#随机选取数据，组成一个256batch的图片，并设置标签
def select_data(path,batch_size = 256):
	img_list = []
	tag_list = []
	for i in range(batch_size):
		file_img = os.listdir(path)
		select_img = random.choice(file_img)
		image = cv2.imread(path+"/"+select_img,cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image,(224,224))
		image = np.array(image, dtype='float32')
		#image = image.transpose((2, 0, 1))
		img_list.append([image])
		tag_list.append(get_tag(select_img))

	if torch.cuda.is_available():
		return torch.tensor(img_list).cuda(),torch.LongTensor(tag_list).cuda()
	else:
		return torch.tensor(img_list),torch.LongTensor(tag_list)


if __name__ == '__main__':
	model = FasterNet([3,6,6,12,12,24,24,48,48,96,96,192,192])
	model = load(model,"test1500.pkl")
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	#使用cuda训练
	if torch.cuda.is_available():
		model = model.cuda()
		loss_fn = loss_fn.cuda()

	right = 0
	totaltime = 0
	for it in range(1500):
		x,y = select_data(path,2)
		a = time.time()
		y_pred = torch.argmax(model(x),dim=1)
		deltatime = time.time()-a
		print("sample:%d,time:%f" % (it,deltatime))
		totaltime += deltatime
		if y_pred[0] == y[0]:
			right +=1
		if y_pred[1] == y[1]:
			right +=1


	print(right/3000,totaltime*1000/3000)
	'''
	for it in range(1500):
		x,y = select_data(path,batch_size)
		y_pred = model(x)
		loss = loss_fn(y_pred,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(it,loss.item())


	save(model,"test1500.pkl")
	'''