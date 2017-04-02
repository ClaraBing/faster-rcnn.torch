require 'nn'
local model = {}
table.insert(model, {'conv1', nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 3, 3)})
table.insert(model, {'relu1', nn.ReLU(true)})
table.insert(model, {'norm1', nn.SpatialCrossMapLRN(3, 0.000050, 0.7500, 1.000000)})
table.insert(model, {'pool1', nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1):ceil()})
table.insert(model, {'conv2', nn.SpatialConvolution(96, 256, 5, 5, 2, 2, 2, 2)})
table.insert(model, {'relu2', nn.ReLU(true)})
table.insert(model, {'norm2', nn.SpatialCrossMapLRN(3, 0.000050, 0.7500, 1.000000)})
table.insert(model, {'pool2', nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1):ceil()})
table.insert(model, {'conv3', nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3', nn.ReLU(true)})
table.insert(model, {'conv4', nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4', nn.ReLU(true)})
table.insert(model, {'conv5', nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5', nn.ReLU(true)})
table.insert(model, {'conv_proposal1', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu_proposal1', nn.ReLU(true)})
-- warning: module 'conv_proposal1_relu_proposal1_0_split [type Split]' not found
table.insert(model, {'proposal_cls_score', nn.SpatialConvolution(256, 18, 1, 1, 1, 1, 0, 0)})
table.insert(model, {'proposal_bbox_pred', nn.SpatialConvolution(256, 36, 1, 1, 1, 1, 0, 0)})
-- warning: module 'proposal_cls_score_reshape [type Reshape]' not found
-- warning: module 'proposal_cls_score_reshape_proposal_cls_score_reshape_0_split [type Split]' not found
-- warning: module 'labels_reshape [type Reshape]' not found
-- warning: module 'labels_reshape_labels_reshape_0_split [type Split]' not found
-- warning: module 'labels_weights_reshape [type Reshape]' not found
return model