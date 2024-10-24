# Yolov3 源码阅读笔记
## Darknet模型
### 构建模型（create_modules）
#### 模型结构与配置
```cfg
[net]  
# Testing  
#batch=1  
#subdivisions=1  
# Training  
batch=16  
subdivisions=1  
width=416  
height=416  
channels=3  
momentum=0.9  
decay=0.0005  
angle=0  
saturation = 1.5  
exposure = 1.5  
hue=.1  
  
learning_rate=0.001  
burn_in=1000  
max_batches = 500200  
policy=steps  
steps=400000,450000  
scales=.1,.1  
  
[convolutional]  
batch_normalize=1  
filters=32  
size=3  
stride=1  
pad=1  
activation=leaky  
  
# Downsample  
  
[convolutional]  
batch_normalize=1  
filters=64  
size=3  
stride=2  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=32  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=64  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
# Downsample  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=3  
stride=2  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=64  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=64  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
# Downsample  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=2  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
# Downsample  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=2  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
# Downsample  
  
[convolutional]  
batch_normalize=1  
filters=1024  
size=3  
stride=2  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=1024  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=1024  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=1024  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=1024  
size=3  
stride=1  
pad=1  
activation=leaky  
  
[shortcut]  
from=-3  
activation=linear  
  
######################  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=1024  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=1024  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=512  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=1024  
activation=leaky  
  
[convolutional]  
size=1  
stride=1  
pad=1  
filters=255  
activation=linear  
  
  
[yolo]  
mask = 6,7,8  
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326  
classes=80  
num=9  
jitter=.3  
ignore_thresh = .7  
truth_thresh = 1  
random=1  
  
  
[route]  
layers = -4  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[upsample]  
stride=2  
  
[route]  
layers = -1, 61  
  
  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=512  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=512  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=256  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=512  
activation=leaky  
  
[convolutional]  
size=1  
stride=1  
pad=1  
filters=255  
activation=linear  
  
  
[yolo]  
mask = 3,4,5  
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326  
classes=80  
num=9  
jitter=.3  
ignore_thresh = .7  
truth_thresh = 1  
random=1  
  
  
  
[route]  
layers = -4  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[upsample]  
stride=2  
  
[route]  
layers = -1, 36  
  
  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=256  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=256  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
filters=128  
size=1  
stride=1  
pad=1  
activation=leaky  
  
[convolutional]  
batch_normalize=1  
size=3  
stride=1  
pad=1  
filters=256  
activation=leaky  
  
[convolutional]  
size=1  
stride=1  
pad=1  
filters=255  
activation=linear  
  
  
[yolo]  
mask = 0,1,2  
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326  
classes=80  
num=9  
jitter=.3  
ignore_thresh = .7  
truth_thresh = 1  
random=1
```


#### 卷积层
```python
# 是否使用归一化，如果需要，则需要加入单独的批量归一化逻辑，与 bias 功能重复
bn = int(module_def["batch_normalize"])
# 卷积核个数
filters = int(module_def["filters"])  
# 卷积核大小
kernel_size = int(module_def["size"])
# 填充
pad = (kernel_size - 1) // 2  
modules.add_module(  
    f"conv_{module_i}",  
    nn.Conv2d(  
        in_channels=output_filters[-1],  
        out_channels=filters,  
        kernel_size=kernel_size,  
        stride=int(module_def["stride"]),  
        padding=pad,  
        bias=not bn,  
    ),  
)  
if bn:
	# 加入批量归一化
    modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))  
# 激活函数
if module_def["activation"] == "leaky":  
    modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
```
#### 上采样层
```python
# 用于将特征图放大
elif module_def["type"] == "upsample":  
    upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")  
    modules.add_module(f"upsample_{module_i}", upsample)
```
#### route层
```python
# 将小尺寸特征图（感受野大）放大和大特征图（感受野小）拼接，用于检测较小物体
elif module_def["type"] == "route": 
    layers = [int(x) for x in module_def["layers"].split(",")]  
    filters = sum([output_filters[1:][i] for i in layers])  
    # shi使用空层占位
    modules.add_module(f"route_{module_i}", EmptyLayer())
```
#### shortcut层
```python
# 残差层，用于更改为训练残差
elif module_def["type"] == "shortcut":  
    filters = output_filters[1:][int(module_def["from"])]  
    # 使用空层占位
    modules.add_module(f"shortcut_{module_i}", EmptyLayer())
```
#### yolo 层
```python
elif module_def["type"] == "yolo":
	# yolov3 会为3种不同尺寸的特征图分别准备3个锚框
    # 选择出适配当前特征图大小的三个锚框 
    anchor_idxs = [int(x) for x in module_def["mask"].split(",")]    
    anchors = [int(x) for x in module_def["anchors"].split(",")]  
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  
    anchors = [anchors[i] for i in anchor_idxs]  
    # 需要检测的类别数量
    num_classes = int(module_def["classes"])  
    # 输入图片的尺寸
    img_size = int(hyperparams["height"])  
    # 添加 yolo 层
    yolo_layer = YOLOLayer(anchors, num_classes, img_size)  
    modules.add_module(f"yolo_{module_i}", yolo_layer)
```

### 向前传播
#### 卷积层、上采样
```python
if module_def["type"] in ["convolutional", "upsample", "maxpool"]:  
	# 调用模块的 forward 函数
    x = module(x)
```

```python
# 上采样 forward 函数
def forward(self, x):  
    x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)  
    return x  
```
#### route 层
```python
elif module_def["type"] == "route":  
	# 将上采样的特征图和前面的特征图拼接进行输出
    x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
```
#### shortcut 层
```python
elif module_def["type"] == "shortcut":  
	# 输出结果加上前面层的输出，让网络学习残差
    layer_i = int(module_def["from"])  
    x = layer_outputs[-1] + layer_outputs[layer_i]
```
#### yolo层（重点）
##### Darknet 向前传播
```python
elif module_def["type"] == "yolo": 
	# 调用 yolo 的 forward 函数，得到输出和损失
    x, layer_loss = module[0](x, targets, img_dim)  
    # 累加损失
    loss += layer_loss  
    # 记录 yolo 层的检测结果
    yolo_outputs.append(x)
```
##### forward
```python
# yolo 层的 forward 函数
def forward(self, x, targets=None, img_dim=None):   
	# x： 输入的特征图
	# targets：groundtrues
	# img_dim: 输入图片的真实尺寸
    self.img_dim = img_dim  
    # batch_size  
    num_samples = x.size(0)  
    # 特征图网格长宽  
    grid_size = x.size(2)  
    
    # 将特征图 reshape 为预测结果的形状，将一个维度拆为 锚框数量 * (xywhc + 检测类别总数)  
    # reshape 后得到的 prediction 结果为
    # num_samples, num_anchors, grid_size, grid_size，num_classes + xywhc
    prediction = (  
        x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)  
        .permute(0, 1, 3, 4, 2)  
        .contiguous()  
    )  
    
    # 通过 sigmoid 函数将预测的 x,y,置信度，类别预测结果归一化
    # x 和 y 表示『预测框的中心点』相对于『当前网格单元』左上角的位置
    # e^w 和 e^h 表示『预测框宽高』相对于『锚框』的比例
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    pred_conf = torch.sigmoid(prediction[..., 4])
    pred_cls = torch.sigmoid(prediction[..., 5:])
      
	if grid_size != self.grid_size:  
		# 计算偏移参数，当特征图大小发生变化后，偏移参数需要重新计算
        self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
  
    #（num_samples, self.num_anchors, grid_size, grid_size，xywh）  
    pred_boxes = FloatTensor(prediction[..., :4].shape)  
    # 预测框中心的 x 坐标，以网格宽度为单位，相对于图像左上角
    pred_boxes[..., 0] = x.data + self.grid_x  
    # 预测框中心的 y 坐标，以网格高度为单位，相对于图像左上角
    pred_boxes[..., 1] = y.data + self.grid_y
    # 预测框的宽度，以网格宽度为单位
    pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  
    # 预测框的高度，以网格高度为单位
    pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h  
  
    # 构建真实图像上输出结果
    output = torch.cat(   
        (  
	        # 乘以网格大小，得到真实图像上的 xywh
            pred_boxes.view(num_samples, -1, 4) * self.stride,  
            pred_conf.view(num_samples, -1, 1),  
            pred_cls.view(num_samples, -1, self.num_classes),  
        ),  
        -1,  
    )  
    if targets is None:
		# 检测任务  
        return output, 0  
    else:  
		# 训练任务
		# ........计算损失..........
        return output, total_loss
```

##### compute_grid_offsets
这个函数用于计算偏移参数，偏移参数可以将相对于
```python
def compute_grid_offsets(self, grid_size, cuda=True):  
    self.grid_size = grid_size  
    g = self.grid_size  
    # 图像大小/特征图大小 = 特征图和原始图像的缩小比例 
    self.stride = self.img_dim / self.grid_size 
     
	# 得到顺序递增的网格，将相对于网格左上角的 xy 转为相对于图像左上角的 xy 
	# 此处的 xy 仍是以网格长宽为单位 1
    self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  
    self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)  
    
    # 得到转换比例后的锚框大小
    # 真实的锚框大小/缩小比例 = 以网格大小为单位的锚框长宽
    self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])  
    self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))  
    self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))  
    print(self.anchor_h.shape)
```
### 损失计算
#### 损失函数
![](https://gitee.com/may1234/md-imgs/raw/master/202410232033409.png)

##### 公式
损失 = 位置误差（正样本） + 置信度误差（正样本误差 + 负样本误差 ）  + 分类误差（正样本）

##### 真实框和预测框
损失计算是针对特征图的**每个网格**上的**每个预测框**，但是每个**真实框只对应一个网格上的一个预测框**，具体来说，一个真实框：
- 对应的网格：真实框中心点位置在特征图中对应的网格
- 对应的预测框：真实框的对应网格中，**『宽高 IOU』 最大的锚框对应的预测框**（预测框预测的 wh 实际上是对应锚框的 wh 的比例）

##### 有无物体（正样本，负样本）
- 真实框对应的预测框视为正样本
- 真实框对应特征图网格中，宽高 IOU 大于阈值的视为**非负样本**，不参与正样本和负样本的损失计算

##### IOU
在 yolov3 中涉及两种 IOU 的计算
- 宽高 IOU（**bbox_wh_iou**）
	用于寻找真实框对应的锚框（三种锚框中选取宽高 IOU 最大的）
- IOU（**bbox_iou**）
	用于计算损失

#### yolo 层 forward 函数（损失计算部分）
```python
iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(  
	pred_boxes=pred_boxes,  
	pred_cls=pred_cls,  
	target=targets,  
	anchors=self.scaled_anchors,  
	ignore_thres=self.ignore_thres,  
)  
# iou_scores：真实值与最匹配的anchor的IOU得分值 class_mask：分类正确的索引  obj_mask：目标框所在位置的最好anchor置为1 noobj_mask obj_mask那里置0，还有计算的iou大于阈值的也置0，其他都为1 tx, ty, tw, th, 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值 tconf 目标置信度  
# Loss : Mask outputs to ignore non-existing objects (except with conf. loss)  
loss_x = self.mse_loss(x[obj_mask], tx[obj_mask]) # 只计算有目标的  
loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])  
loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])  
loss_h = self.mse_loss(h[obj_mask], th[obj_mask])  
loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])   
loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])  
loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj #有物体越接近1越好 没物体的越接近0越好  
loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask]) #分类损失  
total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls #总损失  

# Metrics  
cls_acc = 100 * class_mask[obj_mask].mean()  
conf_obj = pred_conf[obj_mask].mean()  
conf_noobj = pred_conf[noobj_mask].mean()  
conf50 = (pred_conf > 0.5).float()  
iou50 = (iou_scores > 0.5).float()  
iou75 = (iou_scores > 0.75).float()  
detected_mask = conf50 * class_mask * tconf  
precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)  
recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)  
recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)  

self.metrics = {  
	"loss": to_cpu(total_loss).item(),  
	"x": to_cpu(loss_x).item(),  
	"y": to_cpu(loss_y).item(),  
	"w": to_cpu(loss_w).item(),  
	"h": to_cpu(loss_h).item(),  
	"conf": to_cpu(loss_conf).item(),  
	"cls": to_cpu(loss_cls).item(),  
	"cls_acc": to_cpu(cls_acc).item(),  
	"recall50": to_cpu(recall50).item(),  
	"recall75": to_cpu(recall75).item(),  
	"precision": to_cpu(precision).item(),  
	"conf_obj": to_cpu(conf_obj).item(),  
	"conf_noobj": to_cpu(conf_noobj).item(),  
	"grid_size": grid_size,  
}  

return output, total_loss
```
#### build_targets
```python
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):  
	# 用于将groundTrues格式化为方便计算损失的形式  
	# pre_boxes: 预测框信息（batch_size, anchor_num, grid_size,grid_size, 相对于特征图的 xywh）  
	# pred_cls: 预测类型信息（batch_size, anchor_num, grid_size, grid_size, conf）  
	# target: groundTrues (n , batch+class+x+y+w+h)  
	# anchors: (3,2)  
	# ignore_thres: 视为前景的 IOU 阈值
	
	# batchsieze
    nB = pred_boxes.size(0)
    # 每个格子对应了多少个anchor 
    nA = pred_boxes.size(1)
    # 类别的数量   
    nC = pred_cls.size(-1) 
    # grid_size   
    nG = pred_boxes.size(2)   
    
  
    # 这是一个用于表示前景（包含物体）的二进制掩膜，1 为前景， 0 为后景
    # 初始化为 0 是因为大部分为后景，后续只需要将前景设置为 1 
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
     # 这是一个用于表示前景（包含物体）的二进制掩膜，1 为后景， 1 为前景
    # 初始化为 1 是因为大部分为后景，后续只需要将前景设置为 0 
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) 
    # 这是一个用于表示分类是否正确的掩膜  
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0) 
    # 存储每个锚框与真实框的 IoU（Intersection over Union）得分
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0) 

	# 真实框的位置，以网格为单位，相对于网格左上角
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)  

	# 真实框的长宽，以网格为单位
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)   
    th = FloatTensor(nB, nA, nG, nG).fill_(0)  

	# 真实框的类别，one-hot 形式
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)  

	# 真实框的执行度，前景为1，背景为 0
	# tconf = obj_mask.float()

    # 真实框中的 xywh 都是 0-1
    # 需要将其转为以特征图网格为单位, 即 xywh * grid_size
    target_boxes = target[:, 2:6] * nG
    # 截取 xy
    gxy = target_boxes[:, :2]
    # 截取 wh
    gwh = target_boxes[:, 2:]  
    # 计算每个锚框和每个预测框之间的『宽高iou』
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  
    # 计算得到每个真实框对应（iou 最大）的『预测框索引』以及 iou 大小
    best_ious, best_n = ious.max(0)
    
    # 真实框所对应的batch，以及每个框所代表的实际类别 
    b, target_labels = target[:, :2].long().t() 
    # 获取gx，gy,以网格大小为单位，相对于特征图的左上角
    gx, gy = gxy.t()  
	# 获取gw，gh，以网格大小为单位
    gw, gh = gwh.t()  
    # 位置信息，向下取整  
    gi, gj = gxy.long().t()

	# b: 每个真实框所对应的 batch
	# best_n: 每个真实框对应的预测框的下标
	# gi，gj: 每个真实框对应的网格
    # 前景掩膜中实际包含物体的设置成 1，背景掩膜中实际包含物体的设置成 0   
    # 相当于：
    # obj_mask[b, best_n, gj, gi] 可以理解为 torch.stack([obj_mask[b[x], best_n[x], gj[x], gi[x]] for x in range(b.shape[0])])
    obj_mask[b, best_n, gj, gi] = 1  
    noobj_mask[b, best_n, gj, gi] = 0
  
    # 将原先视为背景的，再判断宽高 iou 是否超过阈值，如果超过阈值，则将其视为非负样本，也就是说不参与损失的计算了
    for i, anchor_ious in enumerate(ious.t()): 
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0  
  
  	# 真实框以网格为单位，相对于网格左上角的位置
    tx[b, best_n, gj, gi] = gx - gx.floor() 
    ty[b, best_n, gj, gi] = gy - gy.floor()  
    
    # anchors[best_n][:, 0] 和 anchors[best_n][:, 1] 是真实框对应『锚框』的宽高
    # 1e-16 是一个非常小的数，用来防止除以 0 或 log(0) 导致的数值不稳定
    #『真实框/锚框』
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)  
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)  
    
    # 将真实框的标签转换为one-hot编码形式（对应的类别为 1，其他类别全为 0） 
    tcls[b, best_n, gj, gi, target_labels] = 1 
    
    # 类别预测是否正确，正确为 1，错误为 0
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()  

	# 真实框与对应预测框的 IOU 值
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False） 

	# 置信度，与 obj_mask 一致，正样本为 1，负样本为 0 
    tconf = obj_mask.float()

	# 将 groundtrues 的格式改为了和模型 output 一致
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
```
#### IOU
```python
def bbox_wh_iou(wh1, wh2):  
    wh2 = wh2.t()  
    w1, h1 = wh1[0], wh1[1]  
    w2, h2 = wh2[0], wh2[1]  
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)  
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area  
    return inter_area / union_area  
  
  
def bbox_iou(box1, box2, x1y1x2y2=True):  
    """  
    Returns the IoU of two bounding boxes    """    if not x1y1x2y2:  
        # Transform from center and width to exact coordinates  
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2  
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2  
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2  
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2  
    else:  
        # Get the coordinates of bounding boxes  
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]  
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]  
  
    # get the corrdinates of the intersection rectangle  
    inter_rect_x1 = torch.max(b1_x1, b2_x1)  
    inter_rect_y1 = torch.max(b1_y1, b2_y1)  
    inter_rect_x2 = torch.min(b1_x2, b2_x2)  
    inter_rect_y2 = torch.min(b1_y2, b2_y2)  
    # Intersection area  
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(  
        inter_rect_y2 - inter_rect_y1 + 1, min=0  
    )  
    # Union Area  
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)  
  
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)  
  
    return iou
```
## 训练
