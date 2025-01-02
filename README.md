# Revelio

This repository contains the source code for the paper, **"Revelio: Revealing Important Message Flows in Graph Neural Networks"** by Haoyu He, Isaiah J. King and H. Howie Huang, presented at [ICDE 2025](https://ieee-icde.org/2025/).

![framework](./framework.svg)

All the variables can be found in `config.py`

`*_nc.py` and `*_gc.py` are codes for node classification and graph classification tasks respectively.

---

Here is an example of how this project works.

**1. Train the GNN model**

```
python train_nc.py --dataset cora --model gcn
```

The model will be saved in `./src`.

**2. Explain the model**

```
python run_pygex_nc.py --dataset cora --model gcn --explainer ours
```

The result will be saved in `./res/DATASET_NAME`

**3. Evaluate the explanation performance**

```
python eval_nc.py --dataset cora --model gcn --explainer ours
```
