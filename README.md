# keras-YOLOv3 水印检测模型。在开源的YOLOv3模型基础上，修改部分参数。
## 测试结果（test）
![](https://github.com/yl305237731/yolov3_watermark/blob/master/test/1.png)

## 模型训练

### 1. 数据准备

+ train_image_folder <= 训练图片文件夹.

+ train_annot_folder <= 训练图片标注文件夹，VOC格式.

+ valid_image_folder <= 验证图片文件夹.

+ valid_annot_folder <= 验证图片标注文件夹，VOC格式.
    
其中，图片与标注的名字一一对应，如图片名000001.jpg,对应标注文件名为000001.xml。当验证集不提供时，默认将训练集进行划分，训练集：验证集 = 4:1

### 2. 编辑配置文件
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       288,
        "max_input_size":       448,
        "anchors":              [23,22, 41,89, 46,35, 81,44, 97,31, 101,65, 126,53, 168,37, 242,52],
        "labels":               ["watermark"]   # 指定需要检测的目标类别
    },

    "train": {
        "train_image_folder":   "/home/keras-yolo3/watermark/images/",     # 训练图片路径
        "train_annot_folder":   "/home/keras-yolo3/watermark/annotations/",  # 训练图片对应标注
        "cache_name":           "watermark.pkl",                                       # 生成anchors时会生成
        "pretrained_weights":   "backend.h5",                                    # 预训练权重
        "train_times":          1,                                                     # 每个epoch训练集训练次数
        "batch_size":           8,
        "learning_rate":        1e-4,
        "nb_epochs":            30,
        "warmup_epochs":        3,
        "ignore_thresh":        0.6,                                                   # 低于此阈值，训练时认为box中无目标
        "gpus":                  "1",
        "grid_scales":          [1,1,1],
        "obj_scale":            7,
        "noobj_scale":          1,
        "xywh_scale":           1,
        "class_scale":          1,

        "tensorboard_dir":      "logs",
        "saved_weights_name":   "weights.h5",
        "debug":                true
    } ,

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",
        "cache_name":           ""
    },
    "pred":{
        "obj_thresh":    0.5,          # 预测时包含物体阈值
        "nms_thresh":    0.45,         # 预测时非极大值抑制阈值
        "net_h":         416,
        "net_w":         416,
        "output_mode":    "no_wm" 
}
}
```
labels中包含的目标名将会被作为检测目标

Download pretrained weights for backend at:

https://1drv.ms/u/s!ApLdDEW3ut5fgQXa7GzSlG-mdza6


### 3. 生成acchors

`python gen_anchors.py -c config.json`
将命令行输出的anchors替换config.json配置里的anchors


### 4. 开始训练,后台运行

`nohup python train.py -c config.json &`

当验证集loss在3个连续epoch中未下降时训练停止。

## 检测  
`python predict.py -c config.json -i /image_path/`
默认将检测出带水印结果放到output/下，无水印结果方法non_output/下。

## Evaluation

`python evaluate.py -c config.json`


