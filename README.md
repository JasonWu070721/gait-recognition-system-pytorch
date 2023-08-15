# Gait Recognition System

## Features

- OS: Linux
- Framework: Pytorch, OpenCV
- AI: Convolutional Neural Network, Vision Transformers, Optical flow, OpenPose, Binary Segmentation, triplet-loss
- Language: Python

## Quick start

### Docker

#### Run on Dockerfile

```shell script
docker build -t gait_recognition_system .
```

#### Run on docker-compose

```shell script
docker-compose down

docker-compose up
```

## TODO

- [x] Support PyTorch
- [x] Include DatasetB Database
- [x] Include YOLOv8
- [x] Include openpose
- [ ] Create Binary Segmentation
- [ ] Include NVIDIA Optical Flow
- [ ] Include GaitSet
- [ ] Create test
