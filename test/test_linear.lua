local ts = torch.TestSuite()
local tester = require('test.tester')

require('nn')
require('cunn')

local WeightNorm = require('weightnorm.WeightNorm')

local function generic_learning_test(Linear, type)
  torch.manualSeed(1234)
  cutorch.manualSeed(1234)
  local linear = Linear(10, 1)
  local lwn = WeightNorm.new(linear):type(type)
  local crit = nn.MSECriterion():type(type)

  local lr = 0.1
  for i = 1, 1000 do
    local input = torch.randn(10):type(type)
    local target = input.new(1):fill(input:sum())

    lwn:zeroGradParameters()
    local output = lwn:forward(input)
    local loss = crit:forward(output, target)
    local gradOutput = crit:backward(output, target)
    local gradInput = lwn:backward(input, gradOutput)
    lwn:updateParameters(lr)
  end

  tester:eq(type, torch.type(lwn.output), 'expected output to be of type ' .. type)

  local expected_W = torch.ones(1, 10):type(type)
  local W = (lwn.V / lwn.V:norm(2)) * lwn.g[1]
  tester:eq(W, expected_W, 1e-6)
end

function ts.test_linear_learning()
  generic_learning_test(nn.Linear, 'torch.FloatTensor')
end

function ts.test_linear_learning_cunn()
  generic_learning_test(nn.Linear, 'torch.CudaTensor')
end

return ts
