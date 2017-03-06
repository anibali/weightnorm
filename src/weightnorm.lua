local env = require('weightnorm.env')
require('weightnorm.SpatialConvolution')

setmetatable(env, {
  __call = function(self, module, ...)
    if torch.type(module):find('SpatialConvolution') then
      return self.SpatialConvolution.new(module, ...)
    else
      error('unsupported module type: ' .. torch.type(module))
    end
  end
})

return env
