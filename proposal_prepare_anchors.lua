 -- Translated from expriments/script_VID15.m and
-- functions/rpn/proposal_generate_anchors

-- Invoked by proposal_test.lua

require 'utilities'
require 'matlab_util'
require 'training_utils' -- for 'load_model'

local proposal_calc_output_size
local proposal_generate_anchors

-- local function proposal_prepare_anchors(conf, cache_name, test_net_def_file)
proposal_prepare_anchors = function (conf, cache_name, pnet)
    local output_w_map, output_h_map = proposal_calc_output_size(conf, pnet, w_map_cache, h_map_cache)
    local anchors = proposal_generate_anchors(cache_name)
    return anchors, output_w_map, output_h_map
end


-- local function proposal_calc_output_size(conf, test_net_def_file)
proposal_calc_output_size = function (conf, pnet)
    -- local net = -- TODO: load model?

    local input = torch.range(100, conf.max_pixel_size)
    -- init w/ nan
    local output_w = torch.Tensor(#input):zero() / 0
    local output_h = torch.Tensor(#input):zero() / 0

    -- local softmax = nn.CrossEntropyCriterion():cuda()
    local softmax = nn.LogSoftMax():cuda()

    for i=1,(#input)[1] do
        local s = input[i]
        local imblob = torch.Tensor(3, s, s):zero()
        imblob = imblob:cuda()

        local pnet_out = pnet:forward(imblob) -- TODO: get cls_score?local function proposal_prepare_anchors(conf, cache_name, test_net_def_file)
--        print('\n\npnet_out:')
--        print(pnet_out)
--        print('pnet_out size:')
--        print(pnet_out:size())
        local out = pnet_out[2] -- i.e. select output from conv5 w/ k=3 (same as in proposal_im_detect)
        -- print('\n\nout size:')
        -- print(out:size())
        local cls_score = softmax:forward(out[{{1,2}}])
        if false and i < 5 then
          print('\nout[{{}, {1,2}}] size:')
          print(out[{{},{1,2}}]:size())
          print('\nout[{{1,2}}] size:')
          print(out[{{1,2}}]:size())
          print('\nout[{{}, {1,2}}]:')
          print(out[{{}, {1,2}}])
          print('\ncls_score:')
          print(cls_score)
        end
        output_w[i] = cls_score:size()[1]
        output_h[i] = cls_score:size()[2]
    end

    local output_w_map, output_h_map = {}, {}
    for i=1,(#input)[1] do
        output_w_map[input[i]] = output_w[i]
        output_h_map[input[i]] = output_h[i]
    end
    -- save_obj(w_map_cache, output_w_map)
    -- save_obj(h_map_cache, output_h_map)
    return output_w_map, output_h_map
end

proposal_generate_anchors = function (anchor_cache_file)
    local scales = torch.Tensor({8, 16, 32})
    local ratios = torch.Tensor({0.5, 1, 2})
    local base_size = 16

    if path.exists(anchor_cache_file) then
        return load_obj(anchor_cache_file)
    else
        local base_anchor = torch.Tensor({1, 1, base_size, base_size})
        local ratio_anchors = torch.Tensor({{-7.5, -1.75, 14.5, 8.75},
            {-4, -4, 11, 11}, {-1.5, -7, 8.5, 14}}) -- calculated following ratio_jitter

        -- Modified on Mar 20th: omit scale jitter
        -- ratio & scale jittered
        -- local anchors = torch.Tensor({{-88, -42, 95, 49}, {-180, -88, 187, 95}, 
        --     {-364, -180, 371, 187}, {-60, -60, 67, 67}, {-124, -124, 131, 131},
        --     {-252, -252, 259, 259}, {-40, -84, 47, 91}, {-84, -172, 91, 179},
        --     {-172, -348, 179, 355}})
        save_obj(anchor_cache_file, ratio_anchors)
        return ratio_anchors
    end
end

function main()
    local conf = dofile('config/imagenet_test.lua')
    print('type(conf):')
    print(type(conf))
    print('conf:')
    print(conf)
    print('=== conf end ===')

    local model_path = 'models/new-zf.lua'
    local restore_net = 'output_mine/pnet/imgnet_008000.t7'
    local model = load_model(conf, model_path, restore_net, true)
    -- load models
    -- cfg, model_path, network_filename. true
    -- opt: model = models/new-zf.lua
    --      train = data_mine/ILSVRC2015_VID_test.t7
    --      restore

    local anchor_cache_file = 'cache/rpn_cache/imgnet_test/anchors.t7'
    local w_map_cache_file = 'cache/rpn_cache/imgnet_test/w_map.t7'
    local h_map_cache_file = 'cache/rpn_cache/imgnet_test/h_map.t7'
    local anchors, w_map, h_map =proposal_prepare_anchors(conf, anchor_cache_file, model.pnet, w_map_cache_file, h_map_cache_file)
    save_obj('cache/rpn_cache/imgnet_test/conf.t7', {anchors=anchors, output_width_map=w_map, output_height_map=h_map})
end


-- main()
