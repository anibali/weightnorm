package = "weightnorm"
version = "scm-0"

source = {
  url = "https://github.com/anibali/weightnorm/archive/master.zip",
  dir = "weightnorm-master"
}

description = {
  summary = "Unofficial Torch implementation of weight normalization",
  homepage = "https://github.com/anibali/weightnorm",
  license = "MIT <http://opensource.org/licenses/MIT>"
}

dependencies = {
  "torch >= 7.0"
}

build = {
  type = "builtin",
  modules = {
    ["weightnorm"] = "src/weightnorm.lua",
    ["weightnorm.env"] = "src/weightnorm/env.lua",
    ["weightnorm.WeightNorm"] = "src/weightnorm/WeightNorm.lua"
  }
}
