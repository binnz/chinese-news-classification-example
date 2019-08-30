<h1 align="left">基于BERT的中文新闻分类例子</h1>

This example use chinese news dataset [from here](https://github.com/fate233/toutiao-text-classfication-dataset) to fine tune the bert pretrained model for classification, and save the fine-tuned model to test the result through a rest api deployed by flask. Also the basic thinking come from this blog [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/). Pytorch and tensorflow are both used in this work, especially a library named [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) which help to use pretrained model like BERT, GPT, GPT2 to downstream tasks.

**BERT** is a popular pretrained model [From Google](https://github.com/google-research/bert). Here is some great post for recommend:
>-[illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)

>-[Dissecting BERT Part 1: Understanding the Transformer](https://medium.com/@mromerocalvo/dissecting-bert-part1-6dcf5360b07f)

>-[BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)

<h2 align="center">Getting Started</h2>

#### 1. Download the Pre-trained BERT Model
Download the [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) model and unzip the file

#### 2. Download the dataset followed by the command below and unzip to data dir
```bash
wget https://github.com/fate233/toutiao-text-classfication-dataset/blob/master/toutiao_cat_data.txt.zip
```
#### 3. Prepare the virtual python environment and install the package in requirements.txt

#### 4. Run the command below to fine tune for classification

```bash
python bert_for_classification.py --output_dir your/outout/dir --data_dir toutiao/dataset/dir --data_name toutiao_cat_data.txt --is_add_key_words True
```
#### 4. Set the output file position above to api file, and run the command below to start the flask service

```python
Line 9: model = torch.load('output')
```

```bash
python classification-api.py
```
#### 5. Curl the rest api to test

```bash
curl -X POST http://xx.xx.xx.xx:8000/predict -H 'Content-Type: application/json' -d '{ "text":"珍惜当下 局部新一轮升浪悄然开启" ,"label":"财经"}' |jq
```
```json
    {"Predict Label":"财经 财经","True Label":"财经"}
```

```bash
curl -X POST http://xx.xx.xx.xx:8000/predict -H 'Content-Type: application/json' -d '{ "text":"美国要在亚太建导弹基地？普京：给你脸了是不是！" ,"label":"军事"}' |jq
```
```json
{"Predict Label":"国际 国际","True Label":"军事"}
```



