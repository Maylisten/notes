# Yolo 进化史
## Yolov1
### 网络架构
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281929940.png)

### 损失函数
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281930497.png)

## Yolov2
### Maxpool
删去全连接层，改为通过降采样缩小特征图大小
### Batch Normalization
每次卷积都后面都进行一次 Bath Normalization
### 大分辨率微调
训练结束后额外进行了10次大分辨率图片训练作为微调
### 特征图尺寸
特征图从 7x7 增加为 13x13
### 先验框
yolov2 引入anchor box（先验框），使用 K-Means 聚类算法来根据训练数据生成合理的 5 个 Anchor Boxes 尺寸
### 特征融合
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281946680.png)
## Yolov3

### 纯卷积
没有池化和全连接层
### 多 scale 
3 种 scale（13x13, 26x26,52x52） 的特征图，对应分别有 3 种 scale 的锚框（共 9 种）
不同 scale 的特征图通过上采样（Upsampling） 融合预测
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281954083.png)

### 多标签
softmax 层 改为使用 Sigmoid 激活函数映射到 0-1 之间，支持预测多标签

### 残差连接
缓解梯度消失或梯度爆炸，允许网络层次更深
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281956575.png)

## Yolov4

### Mosaic data augmentation
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281127966.png)

### Self-adversarial-training(SAT)
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281127491.png)

### DropBlock
专门针对卷积层设计，帮助网络在特征提取过程中更好地学习到空间不变性。它的工作原理是在特征图上随机选择一个区域（即一个块）并将该区域内的激活值设置为零，这样网络就不会过于依赖于特定的空间特征位置，进而提升网络的鲁棒性
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281128657.png)

### Label Smoothing
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281128367.png)

### CIOU LOSS
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281126064.png)
### SOFT-NMS(DIOU-NMS)
被抑制的不是直接剔除，而是降低置信度之后，再判断是否剔除

### SPPNet
金字塔式的多尺度池化，将特征图划分成固定数量的网格，然后对每个网格执行池化，保证输出的特征向量维度固定
### CSPNet
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281622530.png)
### CBAM（注意力机制）
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281644841.png)
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281644601.png)

### PAN（Path Aggregation Network）
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281713027.png)

### Mish
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410281714402.png)

## Yolov5
偏向工程，原理变化不大
