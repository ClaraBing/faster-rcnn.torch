require 'nngraph'

-- function create_shared_layers(layers)
function create_proposal_net(layers, prop_layers) 
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
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, conv_steps, stride, LRN, maxPool)
    for i=1,conv_steps do
      ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, stride)
      nInputPlane = nOutputPlane
      dropout = nil -- only one dropout layer per conv-pool block 
    end
    if LRN then
      -- NOTE: not sure whether LRN is within or across channel
      -- size / alpha=0.0001 / beta=0.75 / k=1
      container:add(nn.SpatialCrossMapLRN(3, 0.00005, 0.75))
    end
    if maxPool then
      -- kW / kH / dW / dH / padW / padH
      container:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1):ceil())
    end
    return container
  end  

  local input = nn.Identity()()
  local conv_outputs = {}
  
  local inputs = 3
  local prev = input
  for i,l in ipairs(layers) do
    local net = nn.Sequential()
    ConvPoolBlock(net, inputs, l.filters, l.kW, l.kH, l.padW, l.padH, l.dropout, l.conv_steps, l.stride, l.LRN, l.maxPool)
    inputs = l.filters
    prev = net(prev)
    table.insert(conv_outputs, prev)
  end



-------------------------
--    Proposal Net     --
-------------------------

  -- creates an anchor network which reduces the input first to 256 dimensions 
  -- and then further to the anchor outputs for 3 aspect ratios 
  local function AnchorNetwork(nInputPlane, n, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
    net:add(nn.PReLU())
    -- Modified on Mar 12th: split cls & reg
    -- local branch = nn.ParallelTable()
    net:add(nn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
  end

  local function OutLayers(nInputPlane, n, kernelWidth)
      local cls = nn.Sequential()
      cls:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
      cls:add(nn.PReLU())
      cls:add(nn.SpatialConvolution(n, 9 * 2, 1, 1))

      local reg = nn.Sequential()
      reg:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
      reg:add(nn.PReLU())
      reg:add(nn.SpatialConvolution(n, 9 * 4, 1, 1))

      return cls, reg
  end

  local proposal_outputs = {}
  for i,a in ipairs(prop_layers) do
    table.insert(proposal_outputs, AnchorNetwork(layers[a.input].filters, a.n, a.kW)(conv_outputs[a.input]))
--    local cls, reg = OutLayers(shared_layers[a.input].filters, a.n, a.kW)
--    local cls_out = cls(conv_outputs[a.input])
--    local reg_out = reg(conv_outputs[a.input])
--    table.insert(proposal_outputs, cls_out)
--    table.insert(proposal_outputs, reg_out)
  end
  table.insert(proposal_outputs, conv_outputs[#conv_outputs])
  
    -- create proposal net module, outputs: anchor net outputs followed by last conv-layer output
  local input = nn.Identity()()
  local model = nn.gModule({ input }, proposal_outputs)
  init(model, 'nn.SpatialConvolution')

  -- debug
  print('\n\n=========\npnet ready\n=========\n\n')
  return model
end



---------------------------
--   Classification Net  --
---------------------------

function create_classification_net(shared_layers, prev_input_cnt, class_count, class_layers)
  local conv_outputs = create_shared_layers(shared_layers)
  -- create classifiaction network
  local net = nn.Sequential() 

  for i,l in ipairs(class_layers) do
    net:add(nn.Linear(prev_input_cnt, l.n))
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
  init(model, 'nn.SpatialConvolution')

  -- debug
  print('\n\n=========\ncnet ready\n=========\n\n')
  return model
end


function init(module, name)
  local function init_module(m)
    for k,v in pairs(m:findModules(name)) do
      local n = v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2 / n))
      v.bias:zero()
    end
  end
  module:apply(init_module)
end


function create_model(cfg, shared_layers, prop_layers, class_layers)
  local cnet_ninputs = cfg.roi_pooling.kh * cfg.roi_pooling.kw * shared_layers[#shared_layers].filters
  local model = 
  {
    cfg = cfg,
    layers = shared_layers,
    pnet = create_proposal_net(shared_layers, prop_layers),
    -- cnet = create_classification_net(shared_layers, cnet_ninputs, cfg.class_count + 1, class_layers)
  }
  return model
end
