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

## Run pipeline for big-scale datasets
1. Entering the large-scale directory and download 6 big-scale datasets from the repository of [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale). Notice, you should rename the datasets and place them in the right directory.
```python
cd large-scale
```

2. You can run any models implemented in 'models.py'. For examples, you can run our model on 'genius' dataset by the script:
```python
python main.py --dataset genius --sub_dataset None --method mlpnorm
```
And you can run other models, such as 
```python
python main.py --dataset genius --sub_dataset None --method acmgcn
```
For more experiments running details, you can ref the running sh in the 'experiments/' directory.

3. You can reproduce the experimental results of our method by running the scripts:
```python
bash run_glognn_sota_reproduce_big.sh
bash run_glognn++_sota_reproduce_big.sh
```



## Run pipeline for small-scale datasets
1. Entering the large-scale directory and we provide the original datasets with their splits.
```python
cd small-scale
```

2. You can run our model like the script in the below:
```python
python main.py --no-cuda --model mlp_norm --dataset chameleon --split 0
```
Notice, we run all small-scale datasets on CPUs.
For more experiments running details, you can ref the running sh in the 'sh/' directory.


3. You can reproduce the experimental results of our method by running the scripts:
```python
bash run_glognn_sota_reproduce_small.sh
bash run_glognn++_sota_reproduce_small.sh
```


## Attribution

Parts of this code are based on the following repositories:

- [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

- [PYGCN](https://github.com/tkipf/pygcn)

- [WRGAT](https://github.com/susheels/gnns-and-local-assortativity/tree/main/struc_sim)


## Citation

If you find this code working for you, please cite:

```python
@article{li2022finding,
  title={Finding Global Homophily in Graph Neural Networks When Meeting Heterophily},
  author={Li, Xiang and Zhu, Renyu and Cheng, Yao and Shan, Caihua and Luo, Siqiang and Li, Dongsheng and Qian, Weining},
  journal={arXiv preprint arXiv:2205.07308},
  year={2022}
}
```
