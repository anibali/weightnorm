*Warning: this code has not yet been tested in complete networks, use at own risk*

# Weight normalization

This is an unofficial Torch implementation of ["Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"](http://arxiv.org/abs/1602.07868)
by T. Salimans and D. P. Kingma.

Should work with any weighted layer, including `cudnn` versions.

## Usage

```lua
local wn = require('weightnorm')

-- Construct the neural network as usual, but use `wn()` to wrap weighted
-- layers that you want to be weight normalized.
local net = nn.Sequential()
net:add(wn(nn.Linear(784, 200)))
net:add(nn.ReLU())
net:add(wn(nn.Linear(200, 200)))
net:add(nn.ReLU())
net:add(wn(nn.Linear(200, 10)))

-- [Optional] Perform a data-driven initialization pass.
-- Only works in batch mode.
local batch_input = my_batch_input_loader_function()
net:set_init_pass(true)
net:forward(batch_input)
net:set_init_pass(false)
```

Check out the `test/` folder for more usage examples.
