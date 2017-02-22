require 'nngraph'
require '../utilities'

function create_input_net(l)
  -- Modified on Feb 12th: added "time" for # of input imgs
  local nInputPlane = 3
  local time = 3
  local net = nn.Sequential()

  -- kT = 1 for weight sharing
  net:add(nn.VolumetricConvolution(nInputPlane, l.filters, 1,l.kW,l.kH, 1,l.stride,l.stride, 0,l.padW,l.padH))
  -- VGG style 3x3 convolution building block
  net:add(nn.PReLU():cuda())
  net:add(nn.VolumetricMaxPooling(1,3,3, 1,2,2, 0,1,1):ceil())

  local input = nn.Identity()()
  local prev = net(input)
  local conv_outputs = {}
  table.insert(conv_outputs, prev)
    
  -- create proposal net module, outputs: anchor net outputs followed by last conv-layer output
  local model = nn.gModule({ input }, conv_outputs)
  
  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(model, 'nn.VolumetricConvolution')

  -- debug
  print('\n\n=========\ninput_net ready\n=========\n\n')
  -- print(model)
  save_obj('input_net.obj', model)
  return model
end

function create_proposal_net(layers, anchor_nets)
  -- define  building block functions first

  -- VGG style 3x3 convolution building block
  local function ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, stride)
    container:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH, stride,stride, padW,padH))
    container:add(nn.PReLU())
    if dropout and dropout > 0 then
      container:add(nn.SpatialDropout(dropout))
    end
    return container
  end
  
  -- multiple convolution layers followed by a max-pooling layer
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, conv_steps, stride, maxPool)
    for i=1,conv_steps do
      ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, stride)
      nInputPlane = nOutputPlane
      dropout = nil -- only one dropout layer per conv-pool block 
    end
    if maxPool then
      -- NOTE: not sure whether LRN is within or across channel
      -- size / alpha=0.0001 / beta=0.75 / k=1
      container:add(nn.SpatialCrossMapLRN(3, 0.00005, 0.75))
      -- kW / kH / dW / dH / padW / padH
      container:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1):ceil())
    end
    return container
  end  
  
  -- creates an anchor network which reduces the input first to 256 dimensions 
  -- and then further to the anchor outputs for 3 aspect ratios 
  local function AnchorNetwork(nInputPlane, n, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
    net:add(nn.PReLU())
    net:add(nn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
  end

  local input = nn.Identity()()
    
  local conv_outputs = {}
  
  -- Modified on Feb 12th: 1. change inputs from 3 to 96; 2. added "time" for # of input imgs
  local inputs = 96
  local time = 3
  local prev = input

  -- 1x1 conv to reduce dimension
  local net = nn.Sequential()
  net:add(nn.SpatialConvolution(time*inputs, inputs, 1,1, 1,1, 0,0))
  net:add(nn.PReLU())
  prev = net(prev)
  table.insert(conv_outputs, prev)

  for i,l in ipairs(layers) do
    local net = nn.Sequential()
    ConvPoolBlock(net, inputs, l.filters, l.kW, l.kH, l.padW, l.padH, l.dropout, l.conv_steps, l.stride, l.maxPool)
    inputs = l.filters
    prev = net(prev)
    table.insert(conv_outputs, prev)
    -- debug
    print('pnet: layer ' .. i .. ' ready.')
  end
  
  local proposal_outputs = {}
  for i,a in ipairs(anchor_nets) do
--     print('layers[a.input] (a.input=' .. a.input .. '):')
--     print(layers[a.input])
--     print('conv_outputs[a.input]:')
--     print(conv_outputs[a.input])
    table.insert(proposal_outputs, AnchorNetwork(layers[a.input-1].filters, a.n, a.kW)(conv_outputs[a.input]))
  end
  table.insert(proposal_outputs, conv_outputs[#conv_outputs])
  
    -- create proposal net module, outputs: anchor net outputs followed by last conv-layer output
  local model = nn.gModule({ input }, proposal_outputs)
  
  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(model, 'nn.SpatialConvolution')

  -- debug
  print('\n\n=========\npnet ready\n=========')
  -- print(model)
  print('\n\n')
  save_obj('pnet.obj', 'w')
  return model
end

function create_classification_net(inputs, class_count, class_layers)
  -- create classifiaction network
  local net = nn.Sequential()
  
  local prev_input_count = inputs
  for i,l in ipairs(class_layers) do
    net:add(nn.Linear(prev_input_count, l.n))
    if l.batch_norm then
      net:add(nn.BatchNormalization(l.n))
    end
    net:add(nn.PReLU())
    if l.dropout and l.dropout > 0 then
      net:add(nn.Dropout(l.dropout))
    end
    prev_input_count = l.n
  end
  
  local input = nn.Identity()()
  local node = net(input)
  
  -- now the network splits into regression and classification branches
  
  -- regression output
  local rout = nn.Linear(prev_input_count, 4)(node)
  
  -- classification output
  local cnet = nn.Sequential()
  cnet:add(nn.Linear(prev_input_count, class_count))
  cnet:add(nn.LogSoftMax())
  local cout = cnet(node)
  
  -- create bbox finetuning + classification output
  local model = nn.gModule({ input }, { rout, cout })

  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(model, 'nn.SpatialConvolution')

  -- debug
  print('\n\n=========\ncnet ready\n=========')
  -- print(model)
  print('\n\n')
  save_obj('cnet_3d.obj', model)
  return model
end

function create_model(cfg, input, layers, anchor_nets, class_layers)
  local cnet_ninputs = cfg.roi_pooling.kh * cfg.roi_pooling.kw * layers[#layers].filters
  local model = 
  {
    cfg = cfg,
    layers = layers,
    input_net = create_input_net(input),
    pnet = create_proposal_net(layers, anchor_nets),
    cnet = create_classification_net(cnet_ninputs, cfg.class_count + 1, class_layers)
  }
  return model
end
