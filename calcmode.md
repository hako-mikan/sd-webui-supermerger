## Calculation method

### normal
#### Available modes : All
Normal calculation method. Can be used in all modes.

### cosineA
#### Available modes : weight sum, Add difference

The comparison of two models is performed using cosine similarity, centered on the set ratio, and new ratio is calculated to eliminate loss due to merging. See below for more details.

https://github.com/hako-mikan/sd-webui-supermerger/issues/33
https://github.com/recoilme/losslessmix

### cosineB
#### Available modes : weight sum, Add difference

### smoothAdd
#### Add difference

### tensor
#### Available modes : weight sum only
- This is an Elemental merge that goes beyond Elemental merging.
As you know, each elemental tensor determines the features of an image in U-NET, and in normal merging, the values of each tensor are multiplied by a ratio and added together as shown below (normal). In the tensor method, the tensors are combined by dividing them by the ratio as shown in the figure below (tensor).

! [](https://github.com/hako-mikan/sd-webui-supermerger/blob/images/tensor.jpg)


The tensor size of each element is noted below.
