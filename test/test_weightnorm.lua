local ts = torch.TestSuite()
local tester = require('test.tester')

require('nn')

function ts.test_weightnorm_conv2d()
  local wn = require('weightnorm')

  local conv = nn.SpatialConvolution(3,8, 3,3, 1,1, 1,1)
  local scwn = wn(conv)

  tester:assert(torch.isTypeOf(scwn, 'weightnorm.SpatialConvolution'),
    'expected instance to be an weightnorm.SpatialConvolution')
end

return ts
