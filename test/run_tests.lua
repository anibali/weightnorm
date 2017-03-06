package.path = package.path .. ';./src/?.lua;./src/?/init.lua'

local tester = require('test.tester')

local test_files = {
  'test_conv2d',
  'test_linear',
  'test_misc',
}

for i, test_file in ipairs(test_files) do
  tester:add(require('test.' .. test_file))
end

tester:run()
