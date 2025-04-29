# 基于无约束重力模型和平均增长系数法的交通分布实现

## 1.项目介绍

本项目可以将交通生成阶段预测获得的各校区的发生与吸引量转化为各小区之间的OD分布矩阵。使用了多元回归的方式对对数化的无约束重力模型参数进行标定，而后使用该无约束重力模型进行交通分布预测。当无约束重力模型预测的交通分布无法满足收敛标准时，转用平均增长系数法迭代计算。

## 2.示例
下面是在给定数据上的计算示例
- 交通现状OD分布矩阵
 
| O\D | 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
| 1 | 1000 | 500 | 300 | 200 | 100 |
| 2 | 450 | 1500 | 500 | 100 | 150 |
| 3 | 350 | 200 | 800 | 300 | 50 |
| 4 | 150 | 150 | 300 | 700 | 50 |
| 5 | 80 | 100 | 50 | 70 | 500 |

- 交通现状时间代价矩阵
 
| O\D | 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
| 1 | 5 | 10 | 18 | 20 | 30 |
| 2 | 12 | 4 | 10 | 25 | 20 |
| 3 | 15 | 10 | 8 | 15 | 35 |
| 4 | 20 | 25 | 15 | 10 | 30 |
| 5 | 35 | 25 | 35 | 35 | 12 |

- 交通未来OD总量

| ΣO | 5000 | 4000 | 3000 | 2500 | 2000 |
|-|-|-|-|-|-|
| ΣD | 4000 | 4500 | 3500 | 2500 | 2000 |

- 交通未来时间代价矩阵

| O\D | 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
| 1 | 5 | 10 | 12 | 15 | 20 |
| 2 | 10 | 4 | 8 | 20 | 20 |
| 3 | 10 | 10 | 7 | 10 | 30 |
| 4 | 15 | 20 | 10 | 8 | 20 |
| 5 | 30 | 20 | 30 | 30 | 10 |

采用的无约束重力模型形式如下

对无约束重力模型对数化后，使用交通现状OD矩阵和现状时间代价矩阵的数据对其进行多元线性回归，用于标定无约束重力模型的参数。结果为α=1.0995，β=0.5639，γ=1.1327。

将参数代入后对未来交通OD分布进行迭代预测，满足收敛标准ε<0.003时停止。最终得到结果如下
- 交通未来OD分布矩阵

| O\D | 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
| 1 | 2106 | 1067 | 799 | 583 | 433 |
| 2 | 629 | 1969 | 834 | 282 | 296 |
| 3 | 612 | 679 | 941 | 595 | 178 |
| 4 | 411 | 330 | 665 | 803 | 291 |
| 5 | 253 | 445 | 256 | 234 | 807 |

## 3.使用
将准备好的交通现状OD分布矩阵、交通现状时间代价矩阵、交通未来OD总量、交通未来时间代价矩阵数据填入.csv格式文件中，表格不需要输入行列的标题。


<img src="./result/Figure_1.png">
我们提供了SAM_for_road_disease_b和SAM_for_road_disease_h两个版本的模型，b型版本的模型参数量较小但分割和检测的精度较低，h型版本的模型参数量较大但分割和检测的精度较高，下表展示了两个模型在CrackTree数据集上的表现

| Model | mean_dice | mean_HD95 | mean_IOU | mAP@0.5 | mAP@0.5:0.95 |
|-|-|-|-|-|-|
| SAM_for_road_disease_b | 0.650 | 21.49 | 0.511 | 0.097 | 0.067 |
| SAM_for_road_disease_h | **0.692** | **10.814** | **0.658** | **0.322** | **0.271** |

## 2.使用说明
本项目的开发平台信息如下：
- CentOS 7.9
- Nvidia Tesla T4 16GB
### 2.1 环境配置
在终端中输入以下指令进行环境配置
```bash
conda create -n sam_for_RD python==3.8
```
进入到项目根目录下输入以下指令进行依赖安装
```bash
cd SAM_for_road_disease
pip install -r requirements.txt
```
### 2.2 数据准备
- 模型输入的数据集格式要求为.npz格式文件，每个.npz文件内含两个储存在字典类型中的键值对，键内容为`image`和`label`，值为键对应的图像和标签数据。
- 图像数据为RGB三通道格式，标签数据为单通道的灰度图，背景像素定义为0，目标像素值依类别定义为标签数字值，图像储存为ndarray数据类型。
- 图像大小尺寸无要求。
- 我们提供了代码帮助制作训练和测试所需的数据集，内容请参考`./preprocess/preprocess_data.py`。

### 2.3 训练
将准备好的训练数据集和索引.txt文件放于项目根目录下，sam_vit基础模型要放置于目录`./checkpoints`下。对于sam_vit_b基础模型，在终端中输入以下指令进行训练
```bash
python train.py --warmup --AdamW --root_path <Your training data path> --list_dir <Your list for training indexes> --output <Your output path> 
```
如果要同时训练模型的图像编码器和掩码解码器，输入
```bash
python train.py --warmup --AdamW --root_path <Your training data path> --list_dir <Your list for training indexes> --output <Your output path> --module sam_lora_image_encoder_mask_decoder
```
对于sam_vit_h基础模型，其参数量较大因此需要更多的训练轮次以完成收敛。为获得更快的训练速度和更少的内存占用，我们采用了混合精度等方法进行训练。在终端中输入以下指令进行训练
```bash
python train.py --warmup --AdamW --root_path <Your training data path> --list_dir <Your list for training indexes> --output <Your output path> --tf32 --compile --use_amp
```

### 2.4 测试
将训练好的LoRA放置于模型储存目录`./checkpoints`下，在终端中输入以下指令进行测试
```bash
python test.py --is_savenii --volume_path <Your test dataset path> --output_dir <Your test output directory> --lora_ckpt <path where your LoRA model checkpoints are>
```
如果你同时对图像编码器和掩码解码器进行了微调，请输入下面的指令进行测试
```bash
python test.py --is_savenii --volume_path <Your test dataset path> --output_dir <Your test output directory> --lora_ckpt <path where your LoRA model checkpoints are> --module sam_lora_image_encoder_mask_decoder
```

## 3.分割模型在目标检测任务上的尝试
为了探索分割模型在目标检测任务上拓展应用的可能性，我们还使用目标检测数据集RDD2022_CN对我们的SAM_for_road_disease模型进行了训练和测试。RDD2022_CN数据集的是方框（bounding box）标注而非精确的像素级掩码(mask)标注，这样的标注不符合SAM分割模型的训练输入格式，因此我们依照目标检测的方框标记制作了掩码标注图像，在掩码标注图像中，目标方框内的所有像素值被定义为类别标签值（如裂缝被定义为标签1，孔洞被定义为标签2）
<img src="./materials/label_process.png">
虽然将掩码标注图像中目标方框内的所有像素值被定义为类别标签值的方法轻易地实现了分割模型向目标检测任务上的拓展，但是模型在该数据集上的表现并不优异。这是由于该标注方法实际上属于粗略的像素级标注，其会在训练过程中给模型引入病害周围环境的噪声，导致模型学习到了大量与病害无关的特征，从而导致模型的训练损失一直居高不下极难收敛。对此我们在模型的训练过程中采用了热身和学习率指数衰减策略，即在模型训练初始的一段时间给予其较低的学习率，随着训练的进行，学习率达到一个最大值，而后开始指数衰减，帮助模型的训练损失收敛。下表展示的是两个模型在RDD2022_CN路面病害数据集上的表现

- SAM_for_road_disease_b版本：
  
| Disease category | mean_dice | mean_HD95 | mean_IOU | mAP@0.5 | mAP@0.5:0.95 |
|-|-|-|-|-|-|
| Crack | 0.334 | 106.849 | 0.215 | 0.117 | 0.054 |
| Pothole | 0.164 | **21.487** | 0.145 | 0.161 | 0.054 |
| Patch | **0.339** | 89.170 | **0.313** | **0.289** | **0.154** |
|Average | 0.334 | 99.082 | 0.229 | 0.151 | 0.072 |

- SAM_for_road_disease_h版本：
  
| Disease category | mean_dice | mean_HD95 | mean_IOU | mAP@0.5 | mAP@0.5:0.95 |
|-|-|-|-|-|-|
| Crack | 0.576 | 84.670 | 0.429 | 0.416 | 0.205 |
| Pothole | 0.327 | **23.465** | 0.296 | 0.429 | 0.193 |
| Patch | **0.618** | 64.519 | **0.605** | **0.673** | **0.447** |
| Average | 0.576 | 79.361 | 0.456 | 0.462 | 0.247|

### 4.展望与改进
RDD2022_CN数据集的照片采集视角为远距离全景拍摄，病害在照片中的大小占比只有很小的一部分，属于小目标检测任务。而SAM模型只能输出单一尺度的低分辨率特征图像，无法捕获到病害的细部特征，因此SAM模型在RDD2022_CN数据集上的检测表现较CrackTree差很多。为训练出能更好地适应该小目标检测的任务模型，可以参考CNN中的多尺度金字塔结构，将多个能输出不同分辨率transformer模块进行叠加组合，用于输出图像的多尺度特征。

此外，SAM的visual transformer层采用的是固定分辨率的位置嵌入, 但是模型在测试的时候往往图片的分辨率不是固定的。SAM对此的解决方法是对位置嵌入做双线性插值，而这会损害性能,效率很低而且很不灵活。
因此一种基于视觉transformer的全新架构分割模型segformer被设计了出来，它包含以下特征：
- 分层的金字塔结构，用于提取多重尺度下的目标特征
- 一种新型的卷积位置编码器，避免了不同分辨率输入下的位置插值
- 一个简洁有效的全连接多层感知机解码器

我们同样对该模型进行了微调以使之适应到道路病害分割与检测的任务，该项目内容请见：[segformer_for_crack](https://github.com/HKP-791/Segformer-for-road-disease)

### 5.作者
[Ica_l](desprado233@163.com)
参考项目来源：
```
@article{kirillov2023segany,
  title={[Segment Anything](https://arxiv.org/abs/2304.02643)},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
```
@article{samed,
  title={[Customized Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.13785)},
  author={Kaidong Zhang, and Dong Liu},
  journal={arXiv preprint arXiv:2304.13785},
  year={2023}
}
```