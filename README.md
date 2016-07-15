# Hypernet

## Introduction

This is a short demo of our hyperspectral image classification method based on
convolutional neural networks. Hypernet was initially described in our ACM MM'15
[paper](http://dl.acm.org/citation.cfm?id=2806306) titled Hyperspectral Image
Classification with Convolutional Neural Networks.

The code demonstrates how to train the network for HSI classification on the
Indian Pines dataset while using only 10% of the labeled samples. Alternatively,
a pretrained network is provided for testing.

## Citing Hypernet

If you find Hypernet useful in your research, please consider citing:

@inproceedings{Slavkovikj15Hypernet,
  author =       {Slavkovikj, Viktor and Verstockt, Steven and De Neve, Wesley
                  and Van Hoecke, Sofie and Van de Walle, Rik},
  title =        {{Hyperspectral Image Classification with Convolutional Neural
                  Networks}},
  booktitle =    {{Proceedings of the 23rd ACM International Conference on Multimedia}},
  year =         {2015},
  pages =        {1159--1162},
  numpages =     {4},
}

## Prerequisites

- [Theano](http://deeplearning.net/software/theano/)
- [Lasagne](https://github.com/Lasagne/Lasagne) (tested version: '0.2.dev1')

## Indian Pines dataset

To get the Indian Pines dataset run
```Shell
./data/get_data.sh
```
**Note:** `stats.py` can be used to get more information about the number of
samples used in the training and validation phase.

## Train Hypernet

```Shell
python hypernet.py net1
```

## Test Hypernet

1. Download the pretrained network
```Shell
./trained_models/get_net.sh
```
2. Test the network
```Shell
python hypernet.py ./trained_models/net1.p.gz
```
