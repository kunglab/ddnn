# Embedded Binarized Neural Networks
    
We study embedded Binarized Neural Networks (eBNNs) with the aim of allowing current binarized neural networks (BNNs) in the literature to perform feedforward inference efficiently on small embedded devices. We focus on minimizing the required memory footprint, given that these devices often have memory as small as tens of kilobytes (KB). Beyond minimizing the memory required to store weights, as in a BNN, we show that it is essential to minimize the memory used for temporaries which hold intermediate results between layers in feedforward inference. To accomplish this, eBNN reorders the computation of inference while preserving the original BNN structure, and uses just a single floating-point temporary for the entire neural network. All intermediate results from a layer are stored as binary values, as opposed to floating-points used in current BNN implementations, leading to a 32x reduction in required temporary space. We provide empirical evidence that our proposed eBNN approach allows efficient inference (10s of ms) on devices with severely limited memory (10s of KB). For example, eBNN achieves 95% accuracy on the MNIST dataset running on an Intel Curie with only 15 KB of usable memory with an inference runtime of under 50 ms per sample.

This repository contains a code to train a neural network and generate C/Arduino code for embedded devices such as Arduino 101. 

## Dependencies

This library is dependent on Python 2.7+ and [Chainer](http://chainer.org/). Please install Chainer 1.17+ before starting.

```
pip install chainer
```

## Quick Start
This library has two components: a python module that trains the eBNN and generates a C header file, and the C library which uses the generated header file and is compiled on the target device to perform inference. A quick example of how the python module can be used to train a network is shown below.

```python
import os
import sys
import chainer

from elaas.elaas import Collection
from elaas.family.binary import BinaryFamily
from visualize import visualize
import deepopt.chooser

# Configuration
nepochs = 2 # number of epochs
out_c_file = "simple_convpool.h"

# Setup model type (e.g. Binary)
trainer = Collection(nepochs=nepochs)
trainer.set_model_family(BinaryFamily)

# Get Dataset
train, test = chainer.datasets.get_mnist(ndim=3)
trainer.add_trainset(train)
trainer.add_testset(test)

# Model parameters
trainer.set_searchspace(
    nfilters_embeded=[2],
    nlayers_embeded=[1],
    lr=[1e-3]
)

# Train model
trainer.train()

# generate eBNN C library
data_shape = train._datasets[0].shape[1:]
trainer.generate_c(out_c_file, data_shape)
```

This will generate the simple_convpool.h header file which requires the ebnn.h file located in the c folder. These two files should be included in the C/Arduino code. The C library is used as follows: 

```c
#include <stdio.h>
#include <stdint.h>
#include "simple_convpool.h"

int main()
{
  float input[28*28];
  uint8_t output[1];
   
  //simulate a 28 by 28 greyscale image
  for(int i=0; i < 28*28; ++i) {
    input[i] = i
  }
    
  ebnn_compute(input, output);
  printf("%d\n", output[0]);
   
  return 0;
}
```

For examples of generated networks, see the [c/tests](https://gitlab.com/htkung/ddnn/tree/master/c/tests) folder.

## Paper

Our paper is available [here](http://www.eecs.harvard.edu/~htk/publication/2017-ewsn-mcdanel-teerapittayanon-kung.pdf)

If you use this model or codebase, please cite:
```bibtex
@article{mcdanelembedded,
  title={Embedded Binarized Neural Networks},
  author={McDanel, Bradley and Teerapittayanon, Surat and Kung, HT},
  jornal={Proceedings of the 2017 International Conference on Embedded Wireless Systems and Networks},
  year={2017}
}
```

## License
  
Copyright (c) 2017 Bradley McDanel, Surat Teerapittayanon, HT Kung, Harvard University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.  