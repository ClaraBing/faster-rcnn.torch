-- Translated from Shaoqing Ren's Caffe implementation for faster RCNN:
-- https://github.com/ShaoqingRen/faster_rcnn/functions/rpn/proposal_im_detect.m

-- Invoked by proposal_test.lua

require 'image'
-- require 'matlab_util'
require 'utilities'

function prep_img_for_blob(img, img_means_file_name, target_size, max_size)
    print('in prep_img_for_blob')
    local fin = io.open(img_means_file_name, 'r')
    local content = {}
    for i=1,672 do
        content[i] = fin:read()
        content[i] = content[i]:split(' ')
        for j=1,224 do
            content[i][j] = tonumber(content[i][j])
        end
    end
    fin:close()
    local img_means = torch.Tensor(content):reshape(3, 224, 224)
    print('in prep_img_fore_blob: img_means prepared; img_means size & img_size:')
    print(img_means:size())
    print(img:size())
    if img_means:size() ~= img:size() then
        local img_means_scale = math.max(img:size()[2]/img_means:size()[2], img:size()[3]/img_means:size()[3])
        print('img_means before resize:')
        print(img_means:size())
        img_means = image.scale(img_means, img_means:size()[3]*img_means_scale, img_means:size()[2]*img_means_scale)
        print('img_means after resize:')
        print(img_means:size())
        local y_start = math.floor((img_means:size()[2]-img:size()[2])/2) + 1
        local x_start = math.floor((img_means:size()[3]-img:size()[3])/2) + 1
        print('x_start = ' .. x_start .. ' / y_start = ' .. y_start)
        img_means = img_means[{{}, {y_start, y_start+img:size()[2]-1}, {x_start, x_start+img:size()[3]-1}}]
        print('img_means:size() vs img:size():')
        print(img_means:size())
        print(img:size())
        -- save_obj('im_detect_img_means.t7', img_means)
        -- save_obj('im_detect_img.t7', img)
        img = torch.Tensor(img:double()) - torch.Tensor(img_means) -- bsxfun(torch.csub, img, img_means:float())
    end

    local img_scale = prep_img_for_blob_size(img:size(), target_size, max_size)
    img = image.scale(img, img:size()[3]*img_scale, img:size()[2]*img_scale)
    print('(prep_img_for_blob) img_scale: ' .. img_scale)
    print('(orep_img_for_blob) img size after scaling:')
    print(img:size())

    return img, img_scale
end

function prep_img_for_blob_size(img_size, target_size, max_size)
    local img_size_min, img_size_max
    if img_size[1] > img_size[2] then
        img_size_min, img_size_max = img_size[1], img_size[2]
    else
        img_size_min, img_size_max = img_size[2], img_size[1]
    end
    local img_scale = target_size / img_size_min

    if torch.round(img_scale * img_size_max) > max_size then
        img_scale = max_size / img_size_max
    end
    return img_scale
end

