#### 1.环境配置 pip install -r requirements.txt

#### 2.本实验使用了huggingface的预训练模型bert-base-multilingual-cased,使用时请修改 mult_model.py 中模型存放地址为实际地址

https://huggingface.co/bert-base-multilingual-cased/tree/main

#### 3.请将数据集置于同目录下的data文件夹中

#### 4.运行方式如下:

###### 	--model []   	"选择 model['TEXT', 'IMG', 'MULT']" , 默认为 MULT

###### 	--my_epochs []   	"选择epochs[1, 2, 3, 5, 7, 10, 15]", 默认为 3

###### 	eg. python mult_model.py --model MULT --my_epochs 3	