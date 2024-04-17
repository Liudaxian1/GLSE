# Temporal Knowledge Graph Reasoning via Global and Local Structure-Aware Segment-Evolutionary Representation Learning

This is the released codes of the following paper submitted to TKDD:

Feng Zhao, Kangzheng Liu, Xianzhi Wang, Guandong Xu. Temporal Knowledge Graph Reasoning via Global and Local Structure-Aware Segment-Evolutionary Representation Learning.

## Environment

```shell
python==3.10.9
torch==2.2.1+cu118
dgl==2.1.0+cu118
tqdm==4.66.2
numpy==1.26.4
```

## Introduction

``src``: Python scripts.

``results``: Model files that reproduce the reported results in our paper.

``data``: TKGs used in the experiments.

## Training Command

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 5
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 2
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 2
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 2
```

## Testing Command

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 5 --test
```
```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 2 --test
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 2 --test
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model two --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 2 --test
```