-- Translated from faster_rcnn/functions/rpn/proposal_locate_anchors.m

-- Invoked by proposal_im_detect.lua

require 'matlab_util' -- for meshgrid
require 'prep_img'

local locate_anchors_single_scale

function proposal_locate_anchors(conf, conf_anchors, img_size, target_scale, fm_size)
    if not fm_size then
        fm_size = nil
    end

    local anchors, img_scales
    if target_scale ~= nil then
        anchors, img_scales = locate_anchors_single_scale(img_size, conf, conf_anchors, target_scale, fm_size)
    else
        -- TODO: try out diff scales
        print('ERROR (proposal_locate_anchors): Unspecified target_scale not implemented')
        return
        -- anchors = torch.Tensor()
        -- anchors, img_scales = locate_anchors_single_scale(img)
    end
    return anchors, img_scales
end

locate_anchors_single_scale = function (img_size, conf, conf_anchors, target_scale, fm_size)
    local img_scale, output_size
    if not fm_size then
        img_scale = prep_img_for_blob_size(img_size, target_scale, conf.max_pixel_size)
        img_size = torch.round(torch.Tensor(img_size) * img_scale)
        print('ERROR (proposal_locate_anchors): unspecified fm_size not implemented.')
        return
        -- TODO: need output_w/h_map to calc default val (refer to proposal_local_anchors.m)
        -- output_size = []
    else
        img_scale = prep_img_for_blob_size(img_size, target_scale, conf.max_pixel_size)
        output_size = fm_size
        -- debug
        print('locate_anchors_single_scale: output_size:')
        print(output_size)
    end

    local shift_x = torch.range(0, output_size[2]-1) * conf.feat_stride
    local shift_y =  torch.range(0, output_size[1]-1) * conf.feat_stride
    shift_x, shift_y = meshgrid(shift_x, shift_y)
    -- debug
    print('shift_x size:')
    print(shift_x:size())
    print('shift_y size:')
    print(shift_y:size())
    print('conf_anchors size:')
    print(conf_anchors:size())

    -- shifts_permute: e.g. 1 * 9768 * 4
    local shifts_permute = torch.Tensor(1, shift_x:nElement(), 4):zero()
    shifts_permute[{{1},{},{1}}] = shift_x:transpose(2,1):reshape(shift_x:nElement()) -- expand by concating columns
    shifts_permute[{{1},{},{2}}] = shift_y:transpose(2,1):reshape(shift_y:nElement())
    shifts_permute[{{1},{},{3}}] = shift_x:transpose(2,1):reshape(shift_x:nElement())
    shifts_permute[{{1},{},{4}}] = shift_y:transpose(2,1):reshape(shift_y:nElement())
    -- before: conf_anchors: 3 * 4
    conf_anchors = conf_anchors:reshape(conf_anchors:size()[1], 1, conf_anchors:size()[2])
    -- after: conf_anchors: 3 * 1 * 4
    -- Equivalent to: out_anchors: bsxfun(function(x,y)return x+y end, conf_anchors, shifts_permute)
    local out_anchors = conf_anchors:expand(conf_anchors:size()[1], shifts_permute:size()[2], 4) + shifts_permute:expand(conf_anchors:size()[1], shifts_permute:size()[2], 4)
    print('out_anchors size:')
    print(out_anchors:size())
    -- before: out_anchors: e.g. 3 * 9768 * 4
    out_anchors = out_anchors:transpose(2,1):reshape(out_anchors:nElement()/4, 4):transpose(2,1)
    print('out_anchors size (after reshape):')
    print(out_anchors:size())
    return out_anchors, img_scale
end
