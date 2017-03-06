local env = require('weightnorm.env')
local WeightNorm = require('weightnorm.WeightNorm')

local argcheck = require('argcheck')
require('nn')

-- Convenience method for setting `init_pass` for all weight normization modules
-- in the network. Note: an init pass can only be performed in batch mode.
nn.Module.set_init_pass = argcheck{
  {name = 'self', type = 'nn.Module'},
  {name = 'is_init_pass', type = 'boolean'},
  call =
function(self, is_init_pass)
  self:apply(function(module)
    if torch.isTypeOf(module, WeightNorm) then
      module.init_pass = is_init_pass
    end
  end)
end}

-- Convenience method for applying weight normalization to a weighted layer.
setmetatable(env, {
  __call = function(self, module, ...)
    return self.WeightNorm(module, ...)
  end
})

return env
