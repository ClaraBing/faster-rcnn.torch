require 'nngraph'

nngraph.setDebug(true)

function create_proposal_net(layers, anchor_nets)
  -- define  building block functions first

  -- VGG style 3x3 convolution building block
  local function ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, stride)
    container:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH, stride,stride, padW,padH))
    container:add(nn.ReLU())
    if dropout and dropout > 0 then
      container:add(nn.SpatialDropout(dropout))
    end
    return container
  end
  
  -- multiple convolution layers followed by a max-pooling layer
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, stride, LRN, maxPool)
    ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, stride)
    nInputPlane = nOutputPlane
    dropout = nil -- only one dropout layer per conv-pool block 

    if LRN then
      -- NOTE: not sure whether LRN is within or across channel
      -- size / alpha=0.0001 / beta=0.75 / k=1
      container:add(nn.SpatialCrossMapLRN(3, 0.00005, 0.75))
    end
    if maxPool then
      -- kW / kH / dW / dH / padW / padH
      container:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
    end
    return container
  end  
  
  -- creates an anchor network which reduces the input first to 256 dimensions 
  -- and then further to the anchor outputs for 3 aspect ratios 
  local function AnchorNetwork(nInputPlane, n, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
  end

  local input = nn.Identity()()
    
  local conv_outputs = {}
  
  local inputs = 3
  local time = 3
  local prev = input

  -- 3D Conv for the 1st layer
  -- input image is a 4D tensor: nInputPlane * time(=3) * height * width
  local net = nn.Sequential()
  -- nInputs + nOutputs + kT / kW / kH + dT / dW / dH + padT / padW / padH
  net:add(nn.VolumetricConvolution(inputs, layers[1].filters,  1,layers[1].kW,layers[1].kH,  1,layers[1].stride,layers[1].stride,  0,layers[1].padW,layers[1].padH)) -- kT=1 for weight sharing
  net:add(nn.ReLU())
  net:add(nn.VolumetricMaxPooling(1,3,3, 1,2,2, 0,1,1):ceil())
  inputs = layers[1].filters
  prev = net(prev)
  table.insert(conv_outputs, prev)
  print('pnet: layer 1 (3D conv) ready')

  net = nn.Sequential()
  -- reshape
  -- Note by Bingbin: VarReshape is a self-defined module.
  -- Args: result_n_dim / collapse_dim1 / collapse_dim2
  net:add(nn.VarReshape(3, 1, 2))
  net:add(nn.SpatialConvolution(time*inputs, inputs, 1,1, 1,1, 0,0))
  net:add(nn.ReLU())
  prev = net(prev)
  table.insert(conv_outputs, prev)
  print('pnet: layer 2 (reshape + Spatial Conv) ready')

  for i,l in ipairs(layers) do
    if i ~= 1 then -- skip the 1st layer which has been handled by 3D Conv
      local net = nn.Sequential()
      ConvPoolBlock(net, inputs, l.filters, l.kW, l.kH, l.padW, l.padH, l.dropout,l.stride, l.LRN, l.maxPool)
      inputs = l.filters
      prev = net(prev)
      table.insert(conv_outputs, prev)
      -- debug
      print('pnet: layer ' .. i+1 .. ' ready.')
    end
  end
  -- debug
  print('#conv_outputs: ' .. #conv_outputs)

  local proposal_outputs = {}
  for i,a in ipairs(anchor_nets) do
    print('anchor_id = ' .. i .. ' / a.input = ' .. a.input)
    table.insert(proposal_outputs, AnchorNetwork(layers[a.input].filters, a.n, a.kW)(conv_outputs[a.input+1]))
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
  print('\n\n=========\npnet ready\n=========\n\n')
  return model
end

function create_classification_net(inputs, class_count, class_layers)
  print('cnet inputs = ' .. inputs)
  -- create classifiaction network
  local net = nn.Sequential()
  
  local prev_input_count = inputs
  for i,l in ipairs(class_layers) do
    net:add(nn.Linear(prev_input_count, l.n))
    if l.batch_norm then
      net:add(nn.BatchNormalization(l.n))
    end
    net:add(nn.ReLU())
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
  print('\n\n=========\ncnet ready\n=========\n\n')
  return model
end

function create_model(cfg, layers, anchor_nets, class_layers)
  local cnet_ninputs = cfg.roi_pooling.kh * cfg.roi_pooling.kw * layers[#layers].filters
  local model = 
  {
    cfg = cfg,
    layers = layers,
    pnet = create_proposal_net(layers, anchor_nets),
    cnet = create_classification_net(cnet_ninputs, cfg.class_count + 1, class_layers)
  }
  return model
end
