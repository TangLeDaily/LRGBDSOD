# YHRSOD 

### 数据集

#### DUTS-Train
[http://saliencydetection.net/duts/download/DUTS-TR.zip](http://saliencydetection.net/duts/download/DUTS-TR.zip)
### DUTS-Test
[http://saliencydetection.net/duts/download/DUTS-TE.zip](http://saliencydetection.net/duts/download/DUTS-TE.zip)

- 在根目录创建以下结构中的datasets\train\ 和 atasets\test\

├── YHRSOD \
│ &ensp;&ensp;  └── datasets \
│ &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;      ├── train \
│ &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;      └── DUTS-TR \
│ &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;      └── test \
│ &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;      └── DUTS-TE \
│ &ensp;&ensp;  └── rootmodel \
│ &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; └── ... \
│ &ensp;&ensp;  └── utils \
│ &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; └── ... 

- 将解压后的 *DUTS-TR* 和 *DUTS-TE* 放至上述结构位置

### 训练前配置文件
所有 *train_XXX.py* 文件中的 parser

### 训练

```` 
python train_RCAN.py 
````