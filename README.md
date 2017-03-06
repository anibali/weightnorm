# Weight normalization

This is an unofficial Torch implementation of ["Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"](http://arxiv.org/abs/1602.07868)
by T. Salimans and D. P. Kingma.

## Usage

```
local wn = require('weightnorm')

local net = nn.Sequential()

net:add(wn(nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)))
```

Check out the `test/` folder for more usage examples.

## Features

Features with a check mark are currently implemented.

* [ ] Fully connected layers (`nn.Linear`)
* [x] Spatial convolutions (`nn.SpatialConvolution`, `cudnn.SpatialConvolution`)
