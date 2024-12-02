# yolov8 使用方法
## 检测
### 命令行
```shell
 yolo task=detect mode=predict model=yolov8n.pt source='assets/bus.jpg'
 yolo task=segment mode=predict model=yolov8n-seg.pt source='assets/bus.jpg'
```

### 脚本
```python
from ultralytics import YOLO  
  
model = YOLO("yolov8n.pt")  
  
model.predict(source="assets/bus.jpg")
```