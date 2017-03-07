package.path = package.path .. ';./src/?.lua;./src/?/init.lua'

require('nn')
require('optim')
local tnt = require('torchnet')
local MnistDataset = require('examples.mnist.MnistDataset')
local wn = require('weightnorm')

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1234)

local train_data = MnistDataset.new('data/mnist/train_32x32.t7')
local train_iter = train_data:make_iterator{batch_size = 100, n_threads = 1}

local test_data = MnistDataset.new('data/mnist/test_32x32.t7')
local test_iter = train_data:make_iterator{batch_size = 100, n_threads = 1}

local net = nn.Sequential()
net:add(nn.View(1*32*32))
net:add(wn(nn.Linear(1*32*32, 200)))
net:add(nn.ReLU(true))
net:add(wn(nn.Linear(200, 200)))
net:add(nn.ReLU(true))
net:add(wn(nn.Linear(200, 10)))

local criterion = nn.CrossEntropyCriterion()
local optimiser = {
  method = optim.adam,
  config = {},
  state = {}
}

local params, grad_params = net:getParameters()

local train_loss_meter = tnt.AverageValueMeter()
local test_acc_meter = tnt.ClassErrorMeter({topk = {1}, accuracy = true})

local batch_input
local batch_target
local function do_step()
  grad_params:zero()

  local output = net:forward(batch_input)
  local loss = criterion:forward(output, batch_target)
  local dloss_doutput = criterion:backward(output, batch_target)
  net:backward(batch_input, dloss_doutput)

  train_loss_meter:add(loss)

  return loss, grad_params
end

net:set_init_pass(true)
net:forward(train_iter()().input)
net:set_init_pass(false)

for epoch = 1, 10 do
  train_loss_meter:reset()
  for sample in train_iter() do
    batch_input = sample.input
    batch_target = sample.target

    optimiser.method(
      do_step,
      params,
      optimiser.config,
      optimiser.state
    )
  end

  test_acc_meter:reset()
  for sample in test_iter() do
    local output = net:forward(sample.input)
    test_acc_meter:add(output, sample.target)
  end

  print(string.format('[%2d] train_loss = %0.4f, test_acc = %0.2f%%',
    epoch, train_loss_meter:value(), test_acc_meter:value()[1]))
end
