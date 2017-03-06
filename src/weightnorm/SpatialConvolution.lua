require('nn')
local argcheck = require('argcheck')

local SpatialConvolution, Parent = torch.class('weightnorm.SpatialConvolution', 'nn.Module', require('weightnorm.env'))

SpatialConvolution.__init = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  {name = 'conv', type = 'nn.SpatialConvolution'},
  call =
function(self, conv)
  Parent.__init(self)

  self.conv = conv
  self.V = torch.Tensor(conv.weight:size())
  self.g = torch.Tensor(conv.weight:size(1))
  self.V_grad = torch.Tensor()
  self.g_grad = torch.Tensor()

  self.V:copy(self.conv.weight)
  self.g:fill(1)
end}

SpatialConvolution.updateOutput = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  {name = 'input', type = 'torch.*Tensor'},
  call =
function(self, input)
  if self.init_pass then
    self.conv.bias:zero()
    self.conv.weight:normal(0, 0.05)
  end

  self.V_norm = self.V_norm or self.V.new()
  local V_norm = self.V_norm
  V_norm:resizeAs(self.V):copy(self.V)
  for i = 1, self.V:size(1) do
    V_norm[i]:div(self.V[i]:norm(2))
  end

  local W = self.conv.weight
  W:copy(V_norm)
  for i = 1, self.g:size(1) do
    W[i]:mul(self.g[i])
  end

  local output = self.conv:updateOutput(input)

  if self.init_pass then
    assert(output:dim() == 4, 'expected 4D data (b x f x h x w)')

    local feats = output:size(2)
    local x = output:transpose(1, 2):contiguous():view(feats, -1)
    local mean = x:mean(2):view(feats)
    local var = x:var(2):view(feats)
    local scale_init = torch.rsqrt(var + 1e-8)
    self.g:copy(scale_init)
    torch.cmul(self.conv.bias, -mean, scale_init)
  end

  self.output = output
  return self.output
end}

SpatialConvolution.accGradParameters = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  {name = 'input', type = 'torch.*Tensor'},
  {name = 'gradOutput', type = 'torch.*Tensor'},
  {name = 'scale', type = 'number'},
  call =
function(self, input, gradOutput, scale)
  self.conv:zeroGradParameters()
  self.conv:accGradParameters(input, gradOutput, scale)

  self.g_grad:resizeAs(self.g)
  self.V_grad:resizeAs(self.V)

  local W_grad = self.conv.gradWeight
  for i = 1, self.g:size(1) do
    self.g_grad[i] = torch.dot(W_grad[i], self.V_norm[i])

    local V_norm_val = self.V[i]:norm(2)
    self.V_grad[i]:copy((self.g[i] / V_norm_val) *
      torch.add(W_grad[i], -self.g_grad[i] / V_norm_val, self.V[i]))
  end
end}

SpatialConvolution.parameters = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  call =
function(self)
  return {self.V, self.g, self.conv.bias}, {self.V_grad, self.g_grad, self.conv.gradBias}
end}

SpatialConvolution.type = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  {name = 'type', type = 'string', opt = true},
  {name = 'tensorCache', type = 'table', opt = true},
  call =
function(self, type, tensorCache)
  self.conv:type(type, tensorCache)
  return Parent.type(self, type, tensorCache)
end}

SpatialConvolution.clearState = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  call =
function(self)
  self.conv:clearState()
  nn.utils.clear(self, 'V_norm')
  return Parent.clearState(self)
end}

SpatialConvolution.reset = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  {name = 'stdv', type = 'number', opt = true},
  call =
function(self, stdv)
  self.conv:reset(stdv)
  self.V:copy(self.conv.weight)
  self.g:fill(1)
end}

SpatialConvolution.updateGradInput = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  {name = 'input', type = 'torch.*Tensor'},
  {name = 'gradOutput', type = 'torch.*Tensor'},
  call =
function(self, input, gradOutput)
  self.gradInput = self.conv:updateGradInput(input, gradOutput)
  return self.gradInput
end}

SpatialConvolution.__tostring__ = argcheck{
  {name = 'self', type = 'weightnorm.SpatialConvolution'},
  call =
function(self)
  return string.format('WeightNorm[%s]', self.conv:__tostring__())
end}

return SpatialConvolution
