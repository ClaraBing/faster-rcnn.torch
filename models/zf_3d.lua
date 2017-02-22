require 'models.ZF_model_utilities_3d'

function zf(cfg)
  -- layer here means a block of one or more convolution layers followed by a max-pooling layer
  local input = {filters= 96, kW=7, kH=7, padW=3, padH=3, dropout=0.0, conv_steps=1, stride=2, maxPool=true}

  -- Modified on Jan 24th: +2 parameters for stride & whether to pool
  local layers = { 
    -- Conv1
--    { filters= 96, kW=7, kH=7, padW=3, padH=3, dropout=0.0, conv_steps=1, stride=2, maxPool=true},
    -- Conv2
    { filters=256, kW=5, kH=5, padW=2, padH=2, dropout=0.4, conv_steps=1, stride=2, maxPool=true},
    -- Conv 3, 4
    { filters=384, kW=3, kH=3, padW=1, padH=1, dropout=0.4, conv_steps=2, stride=1, maxPool=false},
    -- Conv 5
    { filters=256, kW=3, kH=3, padW=1, padH=1, dropout=0.4, conv_steps=1, stride=1, maxPool=false};
  }
  
  local anchor_nets = {
    { kW=3, n=256, input=3 },   -- input refers to the 'layer' defined above
    { kW=3, n=256, input=4 },
    { kW=5, n=256, input=4 },
    { kW=7, n=256, input=4 }
  }
  
  local class_layers =  {
    { n=1024, dropout=0.5, batch_norm=true },
    { n=512, dropout=0.5 },
  }
  
  return create_model(cfg, input, layers, anchor_nets, class_layers)
end

return zf
