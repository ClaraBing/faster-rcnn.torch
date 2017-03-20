-- Translated from functions/rpn/proposa_test.m

require 'nms'
require 'utilities'
require 'training_utils'
require 'proposal_prepare_anchors'
require 'proposal_im_detect'

local boxes_filter

local function proposal_test(conf, net_def, net_restore, imdb_file, cache_name)
    if path.exists(cache_name) then
        return load_obj(cache_name)
    else
        -- TODO: mkdir_if_missing
        -- local log_file_base = cache_dir .. ''
        local t_load = os.clock() -- debug
        local model = load_model(conf, net_def, net_restore, true)
        print(string.format('=== check point: model loaded (time = %f) ===', os.clock()-t_load))
        local pnet = model.pnet
        t_load = os.clock() -- debug
        local conf_proposal = load_obj('cache/rpn_cache/imgnet_test/conf.t7')
        -- local conf_anchors, out_w_map, out_h_map = proposal_prepare_anchors(conf, cache_name, pnet)
        print(string.format('=== check point: anchors prepared (time = %f) ===', os.clock()-t_load))
        -- TODO: log files & random seeds
        
        t_load = os.clock() -- debug
        local imdb = load_obj(imdb_file)
        print(string.format('=== check point: imdb loaded (time = %f) ===', os.clock()-t_load))
        local image_ids = keys(imdb.ground_truth)
        local num_imgs = #image_ids
        local aboxes = {} -- torch.Tensor(num_imgs, 1)
        local aboxes_deltas = {} -- torch.Tensor(num_imgs, 1)
        local aanchors = {} -- torch.Tensor(num_imgs, 1)
        local ascores = {} -- torch.Tensor(num_imgs, 1)
        local images = {}

        print('=== check point: about to detect on imgs ===')

        local cnt = 0
        local t_total = os.clock()
        for i=1,num_imgs do
            print('img #' .. i)
            t = os.clock()
            cnt = cnt+1

            local fn = image_ids[i]
            print(fn)
            local status, img = pcall(function () return load_image(fn, conf.color_space, conf.examples_base_path) end)
            print('img loaded')
            if not status then
                print(string.format("Invalid image '%s': %s", fn, img))
                return 0
            end

            local boxes, scores
            boxes, scores, aboxes_deltas[i],aanchors[i], ascores[i] = proposal_im_detect(conf, conf_proposal.anchors, pnet, img)

            print('test img #' .. i .. ': time = ' .. os.clock()-t)

            -- aboxes[i] = torch.cat(boxes, scores, 2)
            aboxes[i] = boxes_filter(boxes, scores, 0.3)
            images[i] = fn

            if i % 50 == 0 then
                save_obj(cache_name, {boxes=aboxes, images=images})
            end
        end -- for i=1, num_img

        save_obj(cache_name, {boxes=aboxes, images=images})
    end -- if cache_dir
end


boxes_filter = function (boxes, scores, nms_overlap_thres, after_nms_topN)
    -- to speed up nms
    -- if per_nma_topN > 0 then
    --    aboxes = aboxes[{{1, math.min(#aboxes, per_nms_topN)},{}}]
    -- end

    -- do nms
    if nms_overlap_thres > 0 and nms_overlap_thres < 1 then
        boxes = nms(boxes, nms_overlap_thres, scores)
    end

    if after_nms_topN and after_nms_topN > 0 then
        boxes = boxes[{{1, math.min(boxes:size()[1], after_nms_topN)},{}}]
    end
    
    return boxes
end


local conf = dofile('config/imagenet_test.lua')
local net_def_file = 'models/new-zf.lua'
local net_restore = 'output_mine/pnet/imgnet_008000.t7'
local imdb_file = 'data_mine/ILSVRC2015_VID_test.t7'
local roidb_cache = 'cache/rpn_cache/imgnet_test/roidb_train.t7'
proposal_test(conf, net_def_file, net_restore, imdb_file, roidb_cache)

