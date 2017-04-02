-- Translated from Shaoqing Ren's Caffe implementation for faster RCNN:
-- https://github.com/ShaoqingRen/faster_rcnn/functions/rpn/proposal_im_detect.m

-- Invoked by proposal_test.lua

require 'image'
require 'matlab_util'
require 'fast_rcnn_bbox_transform_inv'
require 'utilities'
require 'proposal_locate_anchors'
require 'prep_img'
require 'BatchIterator'

local get_image_blob
local filter_boxes
local clip_boxes

local size_debug = true

function proposal_im_detect(conf, conf_anchors, pnet, img)
    -- print('in proposal_im_detect')
    -- local test_scales = 600 -- values from faster_rcnn/functions/rpn/proposal_config.m
    local img_blob, img_scales = get_image_blob(conf, img, conf.test_scales)
    local img_size = img:size()
    local scaled_img_size = torch.round(torch.Tensor({img_size[1], img_size[2], img_size[3]}) * img_scales)

    local lsm = nn.LogSoftMax():cuda()
    -- omitted: steps that permute to fit Caffe mem
    
    -- Get image input
    -- Steps skipped

    local outputs = pnet:forward(img_blob:cuda())
    print('orig outputs:')
    print(outputs)
    outputs = outputs[3] -- i.e. select output from conv5 w/ k=3
    local out_size = outputs:size() -- size = [h, w]

    -- debug
    if size_debug then
        print(string.format('proposal_im_deteect: img_blob size: %s', img_blob:size()))
        print(string.format('proposal_im_detect: img_size: %s', img_size))
        print((string.format('proposal_im_detect: img_scales: %s', img_scales)))
        print('proposal_im_detect: scaled_img_size:')
        print(scaled_img_size)
        print('outputs:')
        print(outputs[{{}, {1,3}, {1,3}}])
        print('proposal_im_detect: outputs size:')
        print(out_size)
    end
    -- reshape outputs to be of size 6 * n, where n is the number of detections: n = 3 * out_size[2] * out_size[3]
    outputs = torch.cat(torch.cat(outputs[{{1,6}, {}, {}}], outputs[{{7,12}, {}, {}}]), outputs[{{13,18}, {}, {}}]):reshape(6,3* out_size[2]*out_size[3])
    -- local box_deltas = torch.cat(torch.cat(outputs[{{3,6}, {}, {}}], outputs[{{9,12}, {}, {}}]), outputs[{{15,18}, {}, {}}]):reshape(4, 3* out_size[2]*out_size[3])

    print('proposal_im_detect: outputs size after reshaping:')
    print(outputs:size())
    local box_deltas = outputs[{{3,6}}]

    -- local fm_size = torch.Tensor({box_deltas:size()[2], box_deltas:size()[1]})
    local fm_size = torch.Tensor({out_size[3], out_size[2]})
    -- omitted: permute & reshape

    local anchors = proposal_locate_anchors(conf, conf_anchors, img_size, conf.test_scales, fm_size)
    local pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas)
    -- TODO: check the type & size of pred_boxes
    local tmp_mult = torch.cdiv(torch.Tensor({img_size[3], img_size[2], img_size[3], img_size[2]})-1, torch.Tensor({scaled_img_size[3], scaled_img_size[2], scaled_img_size[3], scaled_img_size[2]})-1):cuda() 

    if size_debug then
        print('proposal_im_detect: size of tmp_mult:')
        print(tmp_mult:size())
        print('proposal_im_detect: size of pred_boxes:')
        print(pred_boxes:size())
    end

    pred_boxes = torch.cmul(pred_boxes-1, tmp_mult:reshape(1,4):expand(#pred_boxes, 4)) + 1

    if size_debug then
        print('proposal_im_detect: size of pred_boxes (before clip):')
        print(pred_boxes:size())
    end
    
    pred_boxes = clip_boxes(pred_boxes, img:size()[3], img:size()[2])

    -- score: e.g. 2 * 29304
    local scores = lsm:forward(outputs[{{1,2}}])[1] -- [{{},{1}}]

    if size_debug then
        print('scores size:')
        print(scores:size())
    end

    scores = scores:reshape(out_size[2], out_size[3], scores:nElement()/(out_size[2]*out_size[3]))

    -- permute: [w,h,c] -> [c,h,w]
    scores = scores:reshape(scores:nElement())

    local box_deltas_ = deep_copy(box_deltas)
    local anchors_ = deep_copy(anchors)
    local scores_ = deep_copy(scores)

    -- drop too small boxes
    pred_boxes, scores = filter_boxes(conf.test_min_box_size, pred_boxes, scores)
    return pred_boxes, scores, box_deltas_, anchors_, scores__
end

get_image_blob = function (conf, img)
  if type(conf.test_scales) == 'number' or #conf.test_scales == 1 then
    return prep_img_for_blob(img, conf.image_means, conf.test_scales, conf.test_max_size)
  else
    local imgs, img_scales = {}, {}
    for i=1,conf.test_scales do
        imgs[i], img_scales[i] = prep_img_for_blob(img, conf.image_means, conf.test_scales[i], conf.test_max_size)
    end
    local blob = img_list_to_blob(imgs)
    return blob, img_scales
  end
end

-- Ignore boxes that are too small
filter_boxes = function (min_box_size, boxes, scores)
    save_obj('cache/filter_boxes.t7', boxes)
    save_obj('cache/filter_scores.t7', scores)

    -- debug
    print('filter_boxes: boxes size:')
    print(boxes:size())
    print('filter_boxes: scores size:')
    print(scores:size())

    -- boxes: e.g. 29304 * 4
    -- scores: e.g. 29304
    local widths = boxes:select(2,3) - boxes:select(2,1) + 1
    local heights = boxes:select(2,4) - boxes:select(2,2) + 1

    local valid_idx = torch.cmul(torch.gt(widths, min_box_size), torch.gt(heights, min_box_size))
    print('filter_boxes: valid_idx size:')
    print(valid_idx:size())

    -- no boxes pass the filter
    if not valid_idx:any() then
        return -1, -1, -1, -1, -1
    end

    boxes = boxes[valid_idx:reshape(valid_idx:size()[1], 1):expand(valid_idx:size()[1], 4)]
    print('filter_boxes: boxes size after filtering:')
    print(boxes:size())
    boxes = boxes:reshape(boxes:size()[1] / 4, 4)
    print('filter_boxes: boxes size returned:')
    print(boxes:size())
    print('filter_boxes: scores size returned:')
    print(scores[valid_idx]:size())
    return boxes, scores[valid_idx]
end

clip_boxes = function (boxes, img_width, img_height)
-- Clip boxes to be within image boundaries (i.e. x in [1,width] & y in [1,height]])
-- boxes: 2D Tensor
-- img_width/height: numbers (w = img:size()[2] / h = img:size()[1])
    boxes[1] = torch.cmax(torch.cmin(boxes[1], img_width), 1)
    boxes[2] = torch.cmax(torch.cmin(boxes[2], img_height), 1)
    boxes[3] = torch.cmax(torch.cmin(boxes[3], img_width), 1)
    boxes[4] = torch.cmax(torch.cmin(boxes[4], img_height), 1)
    return boxes
end

