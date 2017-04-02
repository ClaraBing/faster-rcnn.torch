require 'utilities'
require 'Rect'

imdb_file = 'data_mine/ILSVRC2015_test_fast_rcnn.t7' 
rois_save_file = 'data_mine/fast_rcnn_gt_rois.t7'

train = load_obj(imdb_file)
img_ids = train.training_set
gt = train.ground_truth

cnt = 0
images = {}
boxes = {}
hash = {}
for i,k in ipairs(img_ids) do
    if not hash[k] then
        v = gt[k]
        cnt = cnt+1
        local seg = v.image_file_name:split('/')
        images[cnt] = seg[7] .. '/' .. seg[8] .. '/' .. seg[9] .. '/' .. seg[10]:split('%.')[1]
        local rois = torch.Tensor(#v.rois, 4):zero()
        for i = 1,#v.rois do
            -- order: [y1 x1 y2 x2]
            rois[{{i},{}}] = torch.Tensor({v.rois[i].rect['minY'], v.rois[i].rect['minX'], v.rois[i].rect['maxY'], v.rois[i].rect['maxX']})
        end
        boxes[cnt] = rois
        hash[k] = true
    end
end
print('cnt: ' .. cnt)

save_obj(rois_save_file, {boxes=boxes, images=images})
