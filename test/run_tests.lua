package.path = package.path .. ';./src/?.lua;./src/?/init.lua'

local tester = require('test.tester')

local test_files = {
  'test_SpatialConvolution',
  'test_weightnorm',
}

for i, test_file in ipairs(test_files) do
  tester:add(require('test.' .. test_file))
end

tester:run()
