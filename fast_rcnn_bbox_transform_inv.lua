require 'matlab_util'
require 'utilities'

function fast_rcnn_bbox_transform_inv(boxes, box_deltas)
-- boxes: conf_anchors
-- box_deltas: box coordinates output from pnet
    if type(boxes) == 'table' then
        print('(fast_rcnn_bbox_transform) boxes: table to tensor')
        boxes = torch.Tensor(boxes)
    end
    boxes = boxes:cuda()
    print(string.format('(fast bbox transform) boxes size: %s', boxes:size()))
    -- size of boxes = 4 * 29304
    local src_w = boxes[3] - boxes[1] + 1 -- i.e. anchor:width() in Anchors.lua
    local src_h = boxes[4] - boxes[2] + 1
    local src_ctr_x = boxes[1] + 0.5 * (src_w-1)
    local src_ctr_y = boxes[2] + 0.5 * (src_h-1)

    if type(box_deltas) == 'table' then
        box_deltas = torch.Tensor(box_deltas)
    end
    box_deltas = box_deltas:cuda()
    print(string.format('(fast bbox transform) box_deltas size: %s', box_deltas:size()))
--     local mask1, mask2, mask3, mask4 = torch.Tensor(box_deltas:size()):zero(), torch.Tensor(box_deltas:size()):zero(), torch.Tensor(box_deltas:size()):zero(), torch.Tensor(box_deltas:size()):zero()
--     for i=1,box_deltas:size()[2]/4 do
--         mask1[{{}, {4*i-3}}] = mask1[{{}, {4*i-3}}] + 1
--         mask2[{{}, {4*i-1}}] = mask2[{{}, {4*i-2}}] + 1
--         mask3[{{}, {4*i-1}}] = mask3[{{}, {4*i-1}}] + 1
--         mask4[{{}, {4*i}}] = mask4[{{}, {4*i}}] + 1
--     end
--     mask1, mask2, mask3, mask4 = mask1:byte(), mask2:byte(), mask3:byte(), mask4:byte()

    local dst_ctr_x = box_deltas[1] -- [mask1]
    local dst_ctr_y = box_deltas[2] -- [mask2]
    local dst_scl_x = box_deltas[3] -- [mask3]
    local dst_scl_y = box_deltas[4] -- [mask4]

    -- debug
    print(string.format('(fast bbox transform) dst_ctr_x size: %s', dst_ctr_x:size()))
    print(string.format('  first line: %s', dst_ctr_x[1]))
    print(string.format('(fast bbox transform) src_w size: %s', src_w:size()))

    local pred_max_x = torch.cmul(dst_ctr_x, src_w) + src_ctr_x
    local pred_max_y = torch.cmul(dst_ctr_y, src_h) + src_ctr_y
    local pred_w = torch.cmul(torch.exp(dst_scl_x), src_w)
    local pred_h = torch.cmul(torch.exp(dst_scl_y), src_h)
    
    -- i.e. x_min / y_min / x_max / y_max
    -- return size = e.g. 4 * 9768
    -- local output = torch.cat(pred_ctr_x - 0.5*(pred_w-1), torch.cat(pred_ctr_y - 0.5*(pred_h-1), torch.cat(pred_ctr_x + 0.5*(pred_w-1), pred_ctr_y + 0.5*(pred_h-1), 2), 2), 2)
    local output = torch.cat(pred_max_x - (pred_w-1), torch.cat(pred_max_y - (pred_h-1), torch.cat(pred_max_x, pred_max_y, 2), 2), 2)
    print('fast bbox transform: output size:')
    print(output:size())
    save_obj('cache/fast_bbox_output.t7', {output=output, box_deltas=box_deltas, pred_max_x=pred_max_x, pred_max_y=pred_max_y, pred_w=pred_w, pred_h=pred_h, boxes=boxes})
    return output
end


