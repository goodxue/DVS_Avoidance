# Python函数说明

## 文件说明 ##

> * `allinc.py, utils2.py`：导入相关的库
> * `DeepCompute.py`：深度估计模块
> * `frontDeal.py`：前处理模块
> * `KalmenFilter.py`：卡尔曼滤波模块
> * `NNTest.py`：神经网络训练模块
> * `models/shuffleNetV2Config.py`：网络基本组元
> * `train_TR.py`：训练相机外部矩阵
> * `CoordinateTranspose.py`：将像素投影到世界坐标
> * `Test_deep&kalmen.py`：本人负责部分的程序（包含均值滤波）

## 各函数说明 ##

1. 外部矩阵逆计算神经网络定义

   ```python
   class SoftPlusModel(nn.Module):
       '''
       param:
       x:torch.tensor([[1.]])
       '''
       return forward(x)
   ```
   
2. 像素坐标投影至世界坐标

   ```python
   def D2toD3(u,v,zc,inner_matrix,outer_matrix):
       '''
       param:
       u:点的像素横坐标
       v:点的像素纵坐标
       inner_matrix:相机内部矩阵的逆
       outer_matrix:相机外部矩阵的广义逆
       xw:点的世界横坐标
       yw:点的世界纵坐标
       zw:点的世界深度坐标，数值上与估计深度zc相同
       '''
       return (xw,yw,zw,1)
   ```

3. 双目深度估计

   ```python
   def deepEstStero(b,w,imgL,imgR):
       '''
       param:
       b:两相机光芯之间的距离
       w:图像中矩形的宽度
       imgL:左侧相机图像矩阵
       imgR:右侧相机图像矩阵
       west:估计物体的宽度
   	'''
       return west
   ```

4. 单目深度估计

   ```python
   def deepEstMono(f,wreal, w):
       '''
       param:
       f:相机的焦距
       wreal:真实物体的大小
       w:图像中矩形的宽度
       deep:估计到的深度
       '''
       return deep
   ```

5. 创建时间滑动窗口

   ```python
   def createTimeWin(eventlist,begin,end):
       '''
       param:
       eventlist:传入的shape为(N,4)的事件列表
       begin:开始计算时间滑动窗口的时间戳
       end:结束计算时间滑动窗口的时间戳
       eventlist_cut:在时间滑动窗口内截取的事件列表,shape:(N,4)
       '''
       return eventlist_cut
   ```

6. 生成Count-Image

   ```python
   def generateCountImage(canvas,eventlist,norm=False,trans=1):
       '''
       param:
       canvas:全0的2D画布矩阵,shape:(w,h)
       eventlist:传入的shape为(N,4)的事件列表
       norm:展示图像时使用，将图像归一化到0-1
       trans:将图像进行转置，方便查看
       countimage:积累了事件的CountImage图像,shape:(w,h)
       '''
       return countimage
   ```

7. 创建时间图像和归一化

   ```python
   def constructTimeImage(listi,canvas):
       '''
       param:
       listi:在每个像素上增加列表，存放时间戳信息
       canvas:积累了事件的CountImage
       return:归一化的事件图像
       '''
       return timeimage
   ```

8. 卡尔曼滤波

   ```python
   def Kalman(Q,R,xmk_1,xk_1,Pmk_1):
       '''
       param:
       Q,R:正态分布，分别表征噪声和测量噪声
       xmk_1:上一时刻传感器测量的状态向量,shape:(6,)
       xk_1:上一时刻由卡尔曼滤波估计出的状态向量,shape:(6,)
       Pmk_1:上一时刻的协方差相机矩阵,shape:(6,6)
       xk:当前时刻由卡尔曼滤波估计出的状态向量
       '''
       return xk
   ```

9. 避障神经网络基本单元定义

   ```python
   class BasicBlock(nn.Module):
       '''
       param:
       inc:输入通道数
       outc:输出通道数
       x:输入/出的张量,torch.tensor
       '''
       return forward(x)
   ```

10. 避障神经网络定义

   ```python
   class FasterNet(nn.Module):
       '''
       param:
       scale:通道缩放比，list类型其中scale[0]为输入通道数，scale[-1]为输出通道数
       nr:输出神经元数
       x:输入/出张量,torch.tensor
       '''
       return forward(x)
   ```

## 注意事项 ##

> * 函数/类定义中存在神经网络模块，这是由于创建归一化时间图用时过长而准备的备用方案，用Python语言中的Pytorch库编写。其中代码行`torch.tensor(x,dtype=torch.float32)`用时为22ms，而`yhat = model(x)`仅用时7ms，而神经网络的权重大小约为200k，经测试，使用MSELoss的情况下，其准确率能达到99%以上。
> * 上述代码仅供参考，部分代码存在缺陷，或需要改进为C++语言编写。本人也正在学习，但是OpenCV环境配置遇到了一些问题，因此先直接上传Python版本的代码。
> * 由于尚未进行相机标定，因此外部矩阵仅能通过数据集中的物体进行估算，因此其存在一定误差，经计算，其坐标中每个轴的平均相对偏差为7.12%左右。