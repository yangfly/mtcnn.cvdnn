## 前言

[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) 是一个快速的人脸检测和对齐算法。本仓库使用 c++ 基于 OpenCV dnn 进行检测。 

![](images/det.jpg)

## 主要环境

- git
- cmake
- VS2017 or VS2015

### 下载源码

```
git clone https://github.com/yangfly/face.cvdnn.git
```

由于 OpenCV-3.4.2 编译库太大，所以请从 [百度云](https://pan.baidu.com/s/1s9P2D23f_cBIQidN6r6RHg) (提取码: hjjb) 下载，然后解压到根目录。

### 构建工程

1. 双击 build.bat 脚本在 build 下生成 mtcnn.sln 工程;
2. 使用 VS2017 打开 mtcnn.sln 工程，右键 `mtcnn` 设为启动项目；
3. 快捷键 `Ctrl + F5` 快速生成 Release X64 版本并运行；
4. 如果要生成 Debug 版本，请将 `工程` → `属性` → `Debug` → `连接器` → `输入` → `附加依赖项` 中的 `opencv_world432.lib` 修改为 `opencv_world432d.lib`。

**注意**：build.bat 脚本默认构建 VS2017 工程，如果要构建 VS2015 工程，请首先将 build.bat 的第一行改为：
```
set VS=2015
```

### 功能函数说明

- `performance`: 多次检测单张图片，打印 CPU 信息和前向速度。
- `demo`：检测单张图片，在窗口中画出检测结果。
- `fddb_detect`：检测 FDDB 数据集，输出标准格式的检测文件用于评估。

### 性能与准确率

##### Win10 CPU

- 测试环境：Win10 Inter(R) Core(TM) i5-4590 @ 3.30GHz
- w|w/o Lnet：在 sample.jpg 上前向 100 次的平均速度，
- Float16：相对于 Float32，预测速度和准确率几乎没有变化，但是模型容量缩小一半。
- FDDB：参数 
- FDDB Time：Fast PNet 相对于 Pnet，通过减少通道数，以节省前向时间。在 FDDB Time 上时间反而长，是因为 Fast Pnet 准确率的下降，导致 Rnet 需要承担更多筛选 False Examples 的工作量，具体而言，`fastP32` 相对于 `mtcnn32`，RNet 的用时占比从 30.57% 上升到了 52.18%。

模式名  |      Float16       |     Fast Pnet      |  w Lnet  | w/o Lnet | FDDB Time | FDDB Disc | FDDB Cont | 误检数
:-----: | :----------------: | :----------------: | :------: | :------: | :-------: | :-------: | :-------: | :----:
mtcnn32 |        :x:         |        :x:         | 20.08 ms | 18.34 ms | 30.08 ms  |   93.00   |   70.02   |  363
mtcnn16 | :heavy_check_mark: |        :x:         | 20.10 ms | 18.65 ms | 29.60 ms  |   92.92   |   69.95   |  363
fastP32 |        :x:         | :heavy_check_mark: | 14.54 ms | 12.48 ms | 39.89 ms  |   90.93   |   68.52   |  268
fastP16 | :heavy_check_mark: | :heavy_check_mark: | 13.74 ms | 12.20 ms | 39.69 ms  |   90.93   |   68.53   |  269


### FDDB 评估

##### 测试参数：
- face_min_size = 40
- face_max_size = 500
- scale_factor = 0.709f
- thresholds = [0.8f, 0.9f, 0.9f]
- precise_landmark = True # with Lnet

![](images/FDDB.jpg)

### 工具

- `shrink_models`: 自动转换所有模型到 float16。
- `performance`：统计在 sample.jpg 上进行 100 次前向的时间。
- `fddb_detect`：检测并生成 FDDB 格式文件，需要 FDDB 数据集。

### 实用建议

将图像存在大量噪声的应用场景中，容易存在大量高分误检，导致进入 Rnet 的 bbox 太多，拖慢 mtcnn 检测器速度，这种时候单纯地提高 Pnet 的 threshold 并不能解决问题。特别感谢 @[Jack Yu](https://github.com/szad670401) 提供了非常实用的两个建议：
1. 对 Pnet 的输入图片进行高斯滤波，缓解噪声；
2. 在 Pnet 每个尺度的 nms 和总体 nms 之前，过滤和其他 bbox 重叠数很少的样本（通常要求重叠数大于 n）。

以上两种建议在代码中都有实现，如果需要分别解注释 `GaussianBlur` 和 `BoxFilter` 即可。


### 参考与致谢

- [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv2/model) @kpzhang93
- [Fast MTCNN](https://github.com/szad670401/Fast-MTCNN) @Jack Yu **推荐：对 Pnet 和 Onet 进行通道缩减，有明显加速效果。**
- [OpenCV](https://github.com/opencv/opencv)