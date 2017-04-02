require 'cunn'
require 'image'
require 'nms'
require 'Anchors'

local Detector_2stage = torch.class('Detector_2stage')
local debug = 0

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
    print('extract_roi')
    local r = localizer:inputToFeatureRect(input_rect)
    local s = feature_layer_output:size()
    r = r:clip(Rect.new(0, 0, s[3], s[2]))
    local idx = { {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX+ 1, r.maxX), r.maxX } }
    return feature_layer_output[idx], idx
end

function Detector_2stage:__init(model)
  local cfg = model.cfg
  self.model = model
  self.anchors = Anchors.new(model.pnet, model.cfg.scales)
  self.localizer = Localizer.new(model.pnet.outnode.children[5])
  self.lsm = nn.LogSoftMax():cuda()
  self.amp = nn.SpatialAdaptiveMaxPooling(cfg.roi_pooling.kw, cfg.roi_pooling.kh):cuda()
end

function Detector_2stage:detect(input, prop_only)
  local pnet = self.model.pnet
  local cnet = self.model.cnet
  local cnet_input_planes = self.model.layers[#self.model.layers].filters
  local amp = self.amp
  local lsm = self.lsm
  
  print('In detect')
  local candidates, outputs = self:get_proposal(self.model.cfg, pnet, input)
  if prop_only then
    return candidates, outputs
  else
    return self:do_classification(cfg, cnet, amp, lsm, cnet_input_planes, outputs, candidates)
  end
end

function Detector_2stage:get_proposal(cfg, pnet, input)
  -- local cfg = self.model.cfg
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local bgclass = cfg.class_count + 1   -- background class
  local cnet_input_planes = self.model.layers[#self.model.layers].filters
  
  local input_size = input:size()
  local input_rect = Rect.new(0, 0, input_size[3], input_size[2])
  
  local t_start
  -- pass image through network
  if debug == 1 then
    t_start = os.clock()
  end
  pnet:evaluate()
  if debug == 1 then
    print(string.format('  eval time: %.3f\n', os.clock()-t_start))
  end
  input = input:cuda()
  if debug == 1 then
    t_start = os.clock()
  end
  print(string.format('input size: %d * %d * %d', input:size()[1], input:size()[2], input:size()[3]))

  local t = os.clock()
  local outputs = pnet:forward(input)
  print(string.format('pnet forward time: %f', os.clock()-t))

  -- print('size of outputs:')
  -- print(outputs:size())
  -- print('#outputs = ' .. #outputs)
  if debug == 1 then
    print(string.format('  pnet forward time: %.3f\n', os.clock()-t_start))
  end

   -- analyse network output for non-background classification
  local matches = {}

  if debug == 1 then
    t_start = os.clock()
  end
  local aspect_ratios = 3
  local result_cnt = 0
  t = os.clock()
  for i=2,3 do
--  for i=5,5 do -- Modified on Mar 7th: only use the last conv layer?
    local layer = outputs[i]
    local layer_size = layer:size()
    if debug == 1 then
      print(layer_size)
    end
    for y=1,layer_size[2] do
      for x=1,layer_size[3] do
        local c = layer[{{}, y, x}]
        if debug == 1 then
          print(c)
        end
        local t_loc = os.clock()
        for a=1,aspect_ratios do

          local ofs = (a-1) * 6
          local cls_out = c[{{ofs + 1, ofs + 2}}] 
          local reg_out = c[{{ofs + 3, ofs + 6}}]
          result_cnt = result_cnt+1
                    
          -- classification
          local c = self.lsm:forward(cls_out)
          -- print('-- time for single lsm: ' .. os.clock()-t_loc)
          t_loc = os.clock()
          --if c[1] > c[2] then
          if math.exp(c[1]) > 0.95 then
            -- regression
            local a = self.anchors:get(i,a,y,x)
            local r = Anchors.anchorToInput(a, reg_out)
            if r:overlaps(input_rect) then
              table.insert(matches, { p=c[1], a=a, r=r, l=i })
            end
          end
          -- print('-- time for process: ' .. os.clock()-t_loc)
        end -- for a=1,aspect_ratios
      end --for x=1,layer_size[3]
    end -- for y=1,layer_size[2]
  end -- for i=1,4
  print(string.format('lsm forward time: %f', os.clock()-t))
  if debug == 1 then
    print(string.format('  non-bg cls time: %.3f\n', os.clock()-t_start))
  end
  

  local winners = {}
  
  if #matches > 0 then
    print('(Detector:detect) #matches > 0')
    
    t = os.clock()
    -- NON-MAXIMUM SUPPRESSION
    local bb = torch.Tensor(#matches, 4)
    local score = torch.Tensor(#matches, 1)
    for i=1,#matches do
      bb[i] = matches[i].r:totensor()
      score[i] = matches[i].p
    end
    
    local iou_threshold = 0.25
    max_score = torch.max(score)
    print('max score: ' .. math.exp(max_score))
    -- Modified on Feb 4th: sort desc by score before nms
    -- sorted_score, idx = torch.sort(score, 1, false)
    -- print(#idx)
    -- sorted_bb = torch.Tensor(#matches, 4)
    -- sorted_matches = {}
    --  for i=1,idx:size()[1] do
    --    sorted_bb[i] = bb[idx[i][1]]
    --    sorted_matches[i] = matches[idx[i][1]]
    --  end
    -- local pick = nms(sorted_bb, iou_threshold, sorted_score)
    -- print('#pick = '..(#pick)[1])
    local pick = nms(bb, iou_threshold, score)
    --local pick = nms(bb, iou_threshold, 'area')
    local candidates = {}
    pick:apply(function (x) table.insert(candidates, matches[x]) end )
    -- pick:apply(function (x) table.insert(candidates, sorted_matches[x]) end )
    print('#pick = '..(#pick)[1])
    -- Mark
    new_max_score = -100
    for i=1,#candidates do
      if candidates[i].p > new_max_score then
        new_max_score = candidates[i].p
      end
      if math.abs(candidates[i].p - max_score) < 0.001 then
        print(candidates[i])
      end
    end
    print(new_max_score)
    print(max_score)
    max_raw = -100
    max_exp = 0
    for i=1, (#pick)[1] do
      max_raw = math.max(max_raw, candidates[i].p)
      max_exp = math.max(max_exp, math.exp(candidates[i].p))
    end
    print(string.format('candidates process time: %f', os.clock()-t))
    
    print('max_raw: ' .. max_raw)
    print('max_exp: ' .. max_exp)
    -- end of modification on Feb 4th
    
    if debug == 1 then
      print(string.format('candidates: %d (before nms: %d)', #candidates, #matches))
    end
    
--     local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
--     for i,v in ipairs(candidates) do
--       -- pass through adaptive max pooling operation
--       local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[5])
--       print('candidates iter #' .. i)
--       print('pi:')
--       print(pi)
--       print('idx:')
--       print(idx)
--       cinput[i] = self.amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
--     end

    -- Mark: need to save sth
    -- return candidates, outputs
    print('result_cnt = ' .. result_cnt)
    return candidates
  end -- if #matches > 0
end -- Detector_2stage:get_proposal





function Detector_2stage:do_classification(cfg, cnet, amp, lsm, cnet_input_planes, outputs, candidates)
  local bgclass = cfg.class_count + 1   -- background class
  -- REGION CLASSIFICATION 
  cnet:evaluate()
  
  -- create cnet input batch
  local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
  for i,v in ipairs(candidates) do
    -- pass through adaptive max pooling operation
    local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[5])
    cinput[i] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
  end
  
  -- send extracted roi-data through classification network
  local coutputs = cnet:forward(cinput)
  local bbox_out = coutputs[1]
  local cls_out = coutputs[2]
  
  local yclass = {}
  for i,x in ipairs(candidates) do
    x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])
    
    local cprob = cls_out[i]
    local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
    
    x.class = c[1]
    x.confidence = p[1]
    if x.class ~= bgclass and debug == 1 then
      print(string.format('  class = %d / score = %.3f\n', x.class, math.exp(x.confidence)))
    end
    if x.class ~= bgclass and math.exp(x.confidence) > 0.2 then
      if not yclass[x.class] then
        yclass[x.class] = {}
      end
      
      table.insert(yclass[x.class], x)
    end
  end

  local winners = {}
  -- run per class NMS
  for i,c in pairs(yclass) do
    -- fill rect tensor
    bb = torch.Tensor(#c, 5)
    for j,r in ipairs(c) do
      bb[{j, {1,4}}] = r.r2:totensor()
      bb[{j, 5}] = r.confidence
    end
    
    pick = nms(bb, 0.1, bb[{{}, 5}])
    pick:apply(function (x) table.insert(winners, c[x]) end ) 
   
  end
  return winners  
end
 
