require 'nngraph'

local function create_model(cfg)
    local input = nn.Identity()()

    -- shared layers
    local conv1 = input - nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 3, 3) - nn.PReLU() - nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1)
    local conv2 = conv1 - nn.SpatialConvolution(96, 256, 5, 5, 2, 2, 2, 2) - nn.PReLU() - nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1)
    local conv3 = conv2 - nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1) - nn.PReLU()
    local conv4 = conv3 - nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1) - nn.PReLU()
    local conv5 = conv4 - nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1) - nn.PReLU()

    -- layer +
    local conv_prop = conv5 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1) - nn.PReLU()

    -- outputs
    local cls = conv_prop - nn.SpatialConvolution(256, 2*9, 1, 1, 1, 1, 1, 1)
    local reg = conv_prop - nn.SpatialConvolution(256, 4*9, 1, 1, 1, 1, 1, 1)

    local pnet = nn.gModule({input}, {cls, reg})
    local layers = {
        { filters= 96, kW=7, kH=7, padW=3, padH=3, dropout=0.0, conv_steps=1, stride=2},
        { filters=256, kW=5, kH=5, padW=2, padH=2, dropout=0.0, conv_steps=1, stride=1},
        { filters=384, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=1, stride=1},
        { filters=384, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=1, stride=1},
        { filters=256, kW=3, kH=3, padW=1, padH=1, dropout=0.0, conv_steps=1, stride=1};
    }

    local model = {cfg=cfg, pnet=pnet, layers=layers}
    return model
end

return create_model
