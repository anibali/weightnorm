local ts = torch.TestSuite()
local tester = require('test.tester')

require('nn')
require('cunn')
require('cudnn')
local image = require('image')

local WeightNorm = require('weightnorm.WeightNorm')

function ts.test_conv2d_instantiate()
  local conv = nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)
  local scwn = WeightNorm.new(conv)

  tester:assert(torch.isTypeOf(scwn, 'nn.Module'), 'expected instance to be an nn.Module')
end

function ts.test_conv2d_clearState()
  local conv = nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)
  local scwn = WeightNorm.new(conv)

  local input = image.lena()
  local output = scwn:forward(input)
  scwn:clearState()

  tester:eq(0, scwn.V_norm:dim(), 'expected V_norm to be an empty tensor')
end

function ts.test_conv2d_forward()
  local conv = nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)
  local scwn = WeightNorm.new(conv)

  local input = image.lena()
  local output = scwn:forward(input)
  tester:eq(output[1]:size(), input[1]:size(), 1e-12)
end

function ts.test_conv2d_backward_zero_grad()
  local conv = nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)
  local scwn = WeightNorm.new(conv)

  local input = image.lena()

  local output = scwn:forward(input)
  local gradOutput = torch.zeros(8, input:size(2), input:size(3))
  local gradInput = scwn:backward(input, gradOutput)
  tester:eq(torch.type(gradInput), torch.type(input), 'expected type of gradInput to match type of input')
  tester:eq(scwn.V_grad:sum(), 0)
  tester:eq(scwn.g_grad:sum(), 0)
  tester:eq(gradInput:sum(), 0)
end

local function generic_learning_test(SpatialConvolution, type)
  torch.manualSeed(1234)
  cutorch.manualSeed(1234)
  local conv = SpatialConvolution(1,1, 3,3, 1,1, 1,1)
  local scwn = WeightNorm.new(conv):type(type)
  local crit = nn.MSECriterion():type(type)

  local lr = 0.1
  for i = 1, 100 do
    local input = torch.randn(1, 32, 32):type(type)
    local target = input:clone()

    scwn:zeroGradParameters()
    local output = scwn:forward(input)
    local loss = crit:forward(output, target)
    local gradOutput = crit:backward(output, target)
    local gradInput = scwn:backward(input, gradOutput)
    scwn:updateParameters(lr)
  end

  tester:eq(type, torch.type(scwn.output), 'expected output to be of type ' .. type)

  local expected_V = torch.zeros(3, 3):type(type)
  expected_V[{2, 2}] = 1
  local V_norm = (scwn.V[{1, 1}] / scwn.V[{1, 1}]:norm(2)):abs()
  tester:eq(V_norm, expected_V, 1e-6, 'expected learnt V to be identity')
end

function ts.test_conv2d_learning()
  generic_learning_test(nn.SpatialConvolution, 'torch.FloatTensor')
end

function ts.test_conv2d_learning_cunn()
  generic_learning_test(nn.SpatialConvolution, 'torch.CudaTensor')
end

function ts.test_conv2d_learning_cudnn()
  generic_learning_test(cudnn.SpatialConvolution, 'torch.CudaTensor')
end

function ts.test_conv2d_init_pass()
  torch.manualSeed(1234)
  cutorch.manualSeed(1234)
  local conv = nn.SpatialConvolution(1,1, 3,3, 1,1, 1,1)
  local scwn = WeightNorm.new(conv)
  local input = torch.randn(1, 1, 32, 32) * 100

  scwn.init_pass = true
  scwn:forward(input)
  scwn.init_pass = false

  local output_after_init = scwn:forward(input)
  tester:eq(output_after_init[{1, 1}]:var(), 1, 1e-6, 'expected unit variance output after init')
  tester:eq(output_after_init[{1, 1}]:mean(), 0, 1e-6, 'expected zero mean output after init')
end

return ts
