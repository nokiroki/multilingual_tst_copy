# [Multilingual Pre-training with Language and Task Adaptation for Multilingual Text Style Transfer (ACL 2022)](http://arxiv.org/abs/2203.08552)
Skoltech NNLP educational project. This repository is not an original official implementation of the work, but a refactored codebase based on the code from https://github.com/laihuiyuan/multilingual-tst.git

Performed within the NNLP coursework at Skoltech.

## Overview

![](./fig/overview.png)
In view of the general scarcity of parallel data, authors proposed a modular approach for multilingual 
formality transfer, which consists of two training strategies that target adaptation to both language and task. 
The approach achieves competitive performance without monolingual task-specific parallel data and can be applied 
to other style transfer tasks as well as to other languages.

## Dataset
- [News-crawl](http://data.statmt.org/news-crawl/): Language-specific generic non-parallel data.
- [RuDetox](https://github.com/skoltech-nlp/russe_detox_2022): Parallel Russian dataset for detoxification task
- [EnDetox](https://github.com/skoltech-nlp/parallel_detoxification_dataset): Parallel English dataset for detoxification task

## Quick Start
To train model you need to put data from [News-crawl](http://data.statmt.org/news-crawl/) in Russian or English to data/news-crawl folder with name train.ru_RU, valid.ru_RU (train.en_XX, valid.en_XX). As train data we use news crawl for 2021 year, as valid data 50k lines from 2020 news crawl for each language.
### Step 1: Language Adaptation Training
Non-parallel data from news-crawl required
```bash
# en_XX, ru_RU
python train_lang_adap.py -dataset news-crawl -lang en_XX
```

### Step 2: Task Adaptation Training
Parallel data from data/detox required -- it's already here
```bash
# en_XX, ru_RU
python train_task_adap.py -dataset detox -lang en_XX
```

### Step 3: Inference

```bash
# ADAPT + EN data (it_IT, fr_XX, pt_XX)
python infer_en_data.py -dataset detox -lang it_IT -style 0 

# ADAPT + EN cross-attn (it_IT, fr_XX, pt_XX)
python infer_en_attn.py -dataset detox -lang it_IT -style 0    
```

## Citation
```
@inproceedings{lai-etal-2022-multi,
    title = "Multilingual pre-training with Language and Task Adaptation for Multilingual Text Style Transfer",
    author = "Lai, Huiyuan  and
      Toral, Antonio  and
      Nissim, Malvina",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = May,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
