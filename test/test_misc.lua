local ts = torch.TestSuite()
local tester = require('test.tester')

local wn = require('weightnorm')

require('nn')
require('cunn')
require('cudnn')
local image = require('image')

function ts.test_misc_helper_function()
  local conv = nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)
  local scwn = wn(conv)

  tester:assert(torch.isTypeOf(scwn, wn.WeightNorm),
    'expected instance to be an weightnorm.WeightNorm')
end

function ts.test_misc_set_init_pass()
  local net = nn.Sequential()
  local scwn = wn(nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1))
  net:add(scwn)

  net:set_init_pass(true)
  tester:eq(scwn.init_pass, true)
  net:set_init_pass(false)
  tester:eq(scwn.init_pass, false)
end

return ts
