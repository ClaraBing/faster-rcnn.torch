require 'models.ZF_model_utilities_prelu'
--require 'models.new-ZF_model_utilities'
--require 'models.rpn_model_utilities'

function zf(cfg)
  -- layer here means a block of one or more convolution layers followed by a max-pooling layer
  -- Modified on Jan 24th: +2 parameters for stride & whether to pool
  local shared_layers = { 
    -- Conv1 + ReLU + LRN + Pooling
    { filters= 96, kW=7, kH=7, padW=1, padH=1, dropout=0.0, stride=2, LRN=true, maxPool=true},
    -- Conv2 + ReLU + LRN + Pooling
    { filters=256, kW=5, kH=5, padW=2, padH=2, dropout=0.0, stride=1, LRN=true, maxPool=true},
    -- Conv 3 + ReLU
    { filters=384, kW=3, kH=3, padW=1, padH=1, dropout=0.0, stride=1, LRN=false, maxPool=false},
    -- Conv 4 + ReLU
    { filters=384, kW=3, kH=3, padW=1, padH=1, dropout=0.0, stride=1, LRN=false, maxPool=false},
    -- Conv 5 + ReLU
    { filters=256, kW=3, kH=3, padW=1, padH=1, dropout=0.0, stride=1, LRN=false, maxPool=false};
  }
  
  local prop_layers = {
    { kW=3, n=256, input=4 },   -- input refers to the 'layer' defined above
    { kW=3, n=256, input=5 },
    { kW=5, n=256, input=5 },
    { kW=7, n=256, input=5 }
  }
--  Modified on Mar 3rd: only 1 AnchorNetworks from the last conv layer
--  local anchor_nets = {
--      { kW=3, n=256, input=5}
--  }
  
  local cls_layers =  {
    -- Modified on Mar 6th: change n from 1024/512 to 4096/4096
    { n=4096, dropout=0.5, batch_norm=true },
    { n=4096, dropout=0.5 },
  }
  
  return create_model(cfg, shared_layers, prop_layers, cls_layers)
end

return zf
