-- require 'detection'
require 'cunn'
require 'nn'
require 'nngraph'
-- local utils = detection.GeneralUtils()
-- To define new models your file should:
-- 1) return the model
-- 2) return a local variable named regressor pointing to the weights of the bbox regressor
-- 3) return a local variable named classifier pointing ro weights of the classifier (without SoftMax!)
-- 4) return the name of the model (used for saving models and logs)
  
 local function create_model(cfg) 
  	local name = 'rpn_zf'
  	backend = backend or nn

	-- SHARED PART
  	local shared   = nn.Sequential()
    -- nn.SpatialConvolution: nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], groups
    local conv1 = nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 3, 3)
	conv1.name = 'conv1'
  	shared:add(conv1)
  	shared:add(nn.PReLU(true))
    -- nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
    shared:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1):ceil())

  	-- Freeze conv2
    local conv2 = nn.SpatialConvolution(96, 256, 5, 5, 2, 2, 2, 2)
  	conv2.name = 'conv2'
	shared:add(conv2)
	shared:add(nn.PReLU(true))
	shared:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1):ceil())

	-- Freeze conv3
    local conv3 = nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1)
	conv3.name = 'conv3'
	shared:add(conv3)
	shared:add(nn.PReLU(true))

	-- Freeze conv4
    local conv4 = nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1)
	conv4.name = 'conv4'
	shared:add(conv4)
	shared:add(nn.PReLU(true))

	-- Freeze conv5
	local conv5 = nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)
	conv5.name = 'conv5'
	shared:add(conv5)
	shared:add(nn.PReLU(true))

--	-- Convolutions and roi info
--	local shared_roi_info = nn.ParallelTable()
--	shared_roi_info:add(shared)
--	shared_roi_info:add(nn.Identity())
	  
--	-- Linear Part
--	local linear = nn.Sequential()
--	linear:add(nn.View(-1):setNumInputDims(3))
--	local fc6 = nn.Linear(25088,4096)
--	fc6.name = 'fc6'
--	linear:add(fc6)
--	linear:add(backend.ReLU(true))
--	linear:add(nn.Dropout(0.5))
--	local fc7 = nn.Linear(4096,4096)
--	fc7.name = 'fc7'
--	linear:add(fc7)
--	linear:add(backend.ReLU(true))
--	linear:add(nn.Dropout(0.5))
  
    -- layer +
    local layer_plus = nn.Sequential()
    
    local conv_prop = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
    conv_prop.name = 'conv_prop'
    layer_plus:add(conv_prop)
    layer_plus:add(nn.PReLU(true))

	-- classifier
	local classifier = nn.SpatialConvolution(256, 2*9, 1, 1, 1, 1, 1, 1)
	classifier.name = 'pnet_classifier'
	-- regressor
	local regressor = nn.SpatialConvolution(256, 4*9, 1, 1, 1, 1, 1, 1)
	regressor.name = 'pnet_regressor'
	local output = nn.ConcatTable()
	-- output:add(classifier)
	-- output:add(regressor)
  
	-- ROI pooling
	-- local ROIPooling = detection.ROIPooling(6,6):setSpatialScale(1/16)

	-- Whole Model
	local model = nn.Sequential()
	-- model:add(shared_roi_info)
    model:add(shared)
	-- model:add(ROIPooling)
	-- model:add(linear)
    model:add(layer_plus)
	model:add(output)

	model:cuda()

    local full_model = 
    {
        cfg = cfg,
        pnet = model
    }
	return full_model,classifier,regressor,name
end

return create_model























