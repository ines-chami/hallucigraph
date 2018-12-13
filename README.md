Graph Convolutional Networks in PyTorch
====
This code was forked from: https://github.com/tkipf/pygcn/tree/master/pygcn
This implementation makes use of the Cora dataset from [2].

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

For link prediction:
```python train_lp.py --model Graphite --save```

For node classification:
```python train_nc.py --model GCN --adj-path [PATH] --adj-thresh 0.95 --adj-weight 0.5```

For joint link prediction and node classification:
```python train_lp_nc.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

```
