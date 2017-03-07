require('nn')
local argcheck = require('argcheck')

local WeightNorm, Parent = torch.class('weightnorm.WeightNorm', 'nn.Module', require('weightnorm.env'))

WeightNorm.__init = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  {name = 'module', type = 'nn.Module'},
  {name = 'weight_feat_dim', type = 'number', default = 1, help = 'dimension in module weights for number of output features'},
  {name = 'output_feat_dim', type = 'number', default = 2, help = 'dimension in module output for number of features'},
  call =
function(self, module, weight_feat_dim, output_feat_dim)
  Parent.__init(self)

  assert(module.weight ~= nil, 'wrapped module must have weights')
  assert(module.bias ~= nil, 'wrapped module must have biases')

  self.module = module
  self.weight_feat_dim = weight_feat_dim
  self.output_feat_dim = output_feat_dim
  self.init_pass = false
  self.V = torch.Tensor(module.weight:size())
  self.g = torch.Tensor(module.weight:size(weight_feat_dim))
  self.V_grad = torch.Tensor():resizeAs(self.V)
  self.g_grad = torch.Tensor():resizeAs(self.g)

  self.V:copy(self.module.weight)
  self.g:fill(1)
end}

WeightNorm.updateOutput = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  {name = 'input', type = 'torch.*Tensor'},
  call =
function(self, input)
  if self.init_pass then
    self.module.bias:zero()
    self.module.weight:normal(0, 0.05)
  end

  local wdim = self.weight_feat_dim
  local odim = self.output_feat_dim

  self.V_norm = self.V_norm or self.V.new()
  local V_norm = self.V_norm
  V_norm:resizeAs(self.V):copy(self.V)
  for i = 1, self.V:size(wdim) do
    V_norm:select(wdim, i):div(self.V:select(wdim, i):norm(2))
  end

  local W = self.module.weight
  W:copy(V_norm)
  for i = 1, self.g:size(1) do
    W:select(wdim, i):mul(self.g[i])
  end

  local output = self.module:updateOutput(input)

  if self.init_pass then
    local feats = output:size(odim)
    local x = output:transpose(1, odim):contiguous():view(feats, -1)
    local mean = x:mean(2):view(feats)
    local var = x:var(2):view(feats)
    local one_on_std = torch.rsqrt(var + 1e-8)
    self.g:copy(one_on_std)
    torch.cmul(self.module.bias, -mean, one_on_std)

    for i = 1, feats do
      output:select(odim, i):add(-mean[i]):mul(one_on_std[i])
    end
  end

  self.output = output
  return self.output
end}

WeightNorm.accGradParameters = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  {name = 'input', type = 'torch.*Tensor'},
  {name = 'gradOutput', type = 'torch.*Tensor'},
  {name = 'scale', type = 'number'},
  call =
function(self, input, gradOutput, scale)
  self.module:zeroGradParameters()
  self.module:accGradParameters(input, gradOutput, scale)

  self.g_grad:resizeAs(self.g)
  self.V_grad:resizeAs(self.V)

  local wdim = self.weight_feat_dim

  local W_grad = self.module.gradWeight
  for i = 1, self.g:size(1) do
    self.g_grad[i] = torch.dot(W_grad:select(wdim, i), self.V_norm:select(wdim, i))

    local V_norm_val = self.V:select(wdim, i):norm(2)
    self.V_grad:select(wdim, i):copy((self.g[i] / V_norm_val) *
      torch.add(W_grad:select(wdim, i), -self.g_grad[i] / V_norm_val, self.V:select(wdim, i)))
  end
end}

WeightNorm.parameters = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  call =
function(self)
  return {self.V, self.g, self.module.bias}, {self.V_grad, self.g_grad, self.module.gradBias}
end}

WeightNorm.type = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  {name = 'type', type = 'string', opt = true},
  {name = 'tensorCache', type = 'table', opt = true},
  call =
function(self, type, tensorCache)
  self.module:type(type, tensorCache)
  return Parent.type(self, type, tensorCache)
end}

WeightNorm.clearState = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  call =
function(self)
  self.module:clearState()
  nn.utils.clear(self, 'V_norm')
  return Parent.clearState(self)
end}

WeightNorm.reset = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  {name = 'stdv', type = 'number', opt = true},
  call =
function(self, stdv)
  self.module:reset(stdv)
  self.V:copy(self.module.weight)
  self.g:fill(1)
end}

WeightNorm.updateGradInput = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  {name = 'input', type = 'torch.*Tensor'},
  {name = 'gradOutput', type = 'torch.*Tensor'},
  call =
function(self, input, gradOutput)
  self.gradInput = self.module:updateGradInput(input, gradOutput)
  return self.gradInput
end}

WeightNorm.__tostring__ = argcheck{
  {name = 'self', type = 'weightnorm.WeightNorm'},
  call =
function(self)
  return string.format('WeightNorm[%s]', self.module:__tostring__())
end}

return WeightNorm
