# 多模态情感分析

 廖泽盛 ECNU

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- numpy==1.19.5
- Pillow==9.1.0
- torch==1.11.0
- transformers==4.19.3

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- result # 记录实验结果
    |-- img_and_text/ # 储存多模态情感分类模型，训练日志和预测结果
    |-- img_only/ # 储存消融实验（只进行图像的情感分类）模型，训练日志和预测结果
    |-- text_only/ # 储存消融实验（只进行文本的情感分类）模型，训练日志和预测结果
|-- config.py # 设置随机种子，以确保在训练和测试过程中的随机性是可重复的
|-- imgClassification.py # 定义了一个图像分类模型ImgModel，使用ViT作为主干网络，通过训练、测试和预测函数对模型进行训练、测试和预测
|-- multiClassification.py # 定义了一个多模态模型MultiModel，结合了图像和文本信息进行分类任务，使用ViT和BERT作为主干网络，通过训练、测试和预测函数对模型进行训练、测试和预测
|-- multiDataset.py # 定义了一个多模态数据集类MultiDataset，用于加载包含图像和文本信息的数据，并进行预处理，包括文本的编码和图像的提取特征
|-- requirements.txt # 需要安装的依赖
|-- run.py # 运行
|-- runUtil.py # 定义辅助函数的训练、测试、预测的流程
|-- textClassification.py # 定义了一个文本分类模型TextModel，使用BERT作为主干网络，通过训练、测试和预测函数对模型进行训练、测试和预测
```

## Run 

常规实验

1. 训练，并在训练中进行验证数据的指标计算：
```powershell
python run.py –train –test –mode img_and_text
```

2. 训练好的模型对无标签的测试数据进行预测：
```powershell
python run.py  --predict --mode img_and_text --prediction_path result/img_and_text/prediction.txt
```
消融实验

只进行图像的情感分类：

3. 训练

```powershell
python run.py --train --test --mode img_only --cache_model_path cache/img_only_model
```
4. 训练好的模型对无标签的测试数据进行预测：

```python
bash run_glognn_sota_reproduce_big.sh
bash run_glognn++_sota_reproduce_big.sh
```

只进行文本的情感分类：

5. 训练

```powershell
python run.py --train --test --mode text_only --cache_model_path cache/text_only_model
```

6. 训练好的模型对无标签的测试数据进行预测：

```powershell
python run.py --mode text_only --predict --prediction_path result/text_only/prediction.txt --cache_model_path cache/text_only_model
```




## Attribution

Parts of this code are based on the following repositories:

- NAN


## Citation

If you find this code working for you, please cite:

```python
@article{
    NAN
  author={liao},
  year={2023}
}
```
