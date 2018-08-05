# CascadeNet in Keras
“CascadeNet: Modified ResNet with Cascade Blocks”
In this paper, we present an effective deep CNN architecture modified on the typical Residual Network (ResNet), named as Cascade Network
(CascadeNet), by repeating cascade building blocks. Each cascade block contains independent convolution paths to pass information in the previous layer and the middle one. This strategy exposes a concept of “cross-passing” which differs from the ResNet that stacks simple building blocks with residual connections.


## Setup
Download and extract CIFAR-10 data. 

## Training and Testing
For training:
```
   python train.py --mode 0
```

For testing:
```
  python train.py --mode 1
```

## Updating...

## Acknowledgment
Xiang Li, W. Li, Qian Du, "CascadeNet: Modified ResNet with Cascade Blocks,"
