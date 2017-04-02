require 'cunn'
require 'BatchIterator'
require 'Localizer'

function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects, 
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(0, 0, s[3], s[2]))
  local idx = { {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX + 1, r.maxX), r.maxX } }
  return feature_layer_output[idx], idx
end

function create_objective_rpn(model, weights, gradient, batch_iterator, stats)
  local cfg = model.cfg
  local pnet = model.pnet
-- Mark: comment out cnet part
-- local cnet = model.cnet 
--  local bgclass = cfg.class_count + 1   -- background class

  local anchors = batch_iterator.anchors    
  local localizer = Localizer.new(pnet.outnode.children[5])
    
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cnll = nn.ClassNLLCriterion():cuda() -- Note by Bingbin: Negative Log Likelihood
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  smoothL1.sizeAverage = false
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local cnet_input_planes = model.layers[#model.layers].filters
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()
  
  local function cleanAnchors(examples, outputs)
    local cls_out, reg_out = outputs[1], outputs[2]
    local i = 1
    while i <= #examples do
      local anchor = examples[i][1]
      local fmSize = cls_out[anchor.layer]:size()
      -- Note by Bingbin: remove across-boundary anchors
      -- Modified on Mar 13th: fmSize index changed since fm changed to 2D
      if anchor.index[2] > fmSize[1] or anchor.index[3] > fmSize[2] then
        print(string.format('cleanAnchors: exceed -> crop: anchor: %d * %d / fm: %d * %d', anchor.index[2], anchor.index[3], fmSize[1], fmSize[2]))
        table.remove(examples, i)   -- accessing would cause ouf of range exception
      else
        i = i + 1
      end 
    end
  end
  
  local function lossAndGradient(w)
      if w ~= weights then
        weights:copy(w)
      end
      gradient:zero()

      -- statistics for proposal stage      
      local cls_loss, reg_loss = 0, 0
      local cls_count, reg_count = 0, 0
      local delta_outputs = {}
      -- Modified on Mar 3rd: calculate fg/bg accuracy
      local pos_count, neg_count = 0, 0
      local pfg_accu, pbg_accu = 0, 0
   
      -- statistics for fine-tuning and classification stage
      -- local creg_loss, creg_count = 0, 0
      -- local ccls_loss, ccls_count = 0, 0
      
      -- enable dropouts 
      pnet:training()
      -- cnet:training()
      
      t_batch_start = os.clock()
      local batch = batch_iterator:nextTraining()
      local debug_out = true
      for i,x in ipairs(batch) do
        local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
        local p = x.positive        -- get positive and negative anchors examples
        local n = x.negative

        -- debug
        print('img size:')
        print(img:size())
        -- run forward convolution
        local outputs = pnet:forward(img)
        -- debug
--        if (debug_out and i<3) then
--          print('outputs:')
--          print(outputs)
--            j = 1
--            print('outputs[' .. j .. '][1:5, 1:7, 1:7]')
--            print(outputs[j][{{1,5}, {1,7}, {1,7}}])
--            print('\n')
--        end
        
        -- debug
        print('before cleaning Anchors: #p=' .. #p .. ' / #n=' .. #n)
        -- ensure all example anchors lie withing existing feature planes 
        cleanAnchors(p, outputs)
        cleanAnchors(n, outputs)
        print('after cleaning Anchors: #p=' .. #p .. ' / #n=' .. #n)
        
        -- clear delta values for each new image
        for i,out in ipairs(outputs) do
          if not delta_outputs[i] then
            delta_outputs[i] = torch.FloatTensor():cuda()
          end
          delta_outputs[i]:resizeAs(out)
          delta_outputs[i]:zero()
        end
        
        local roi_pool_state = {}
        local input_size = img:size()
        -- local cnetgrad
       
        -- process positive set
        for i,x in ipairs(p) do
          local anchor = x[1]
          local roi = x[2]
          local l = anchor.layer
          print('anchor-layer = ' .. l)
          
          local out = outputs[l]
          local delta_out = delta_outputs[l]
           
          local idx = anchor.index
          local v = out[idx]
          local d = delta_out[idx]
            
          -- classification
          cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
          local dc = softmax:backward(v[{{1, 2}}], 1)
          if (debug_out and i<10) then
            print('v:')
            print(v)
            print('\n\ndc:')
            print(dc)
            print('\n\nd[{{1,2}}]:')
            print(d[{{1,2}}])
          end
          d[{{1,2}}]:add(dc)

          if v[1] > v[2] then pfg_accu = pfg_accu+1; end
          
          -- box regression
          local reg_out = v[{{3, 6}}]
          local reg_target = Anchors.inputToAnchor(anchor, roi.rect):cuda()  -- regression target
          local reg_proposal = Anchors.anchorToInput(anchor, reg_out)
          reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * 10
          local dr = smoothL1:backward(reg_out, reg_target) * 10
          d[{{3,6}}]:add(dr)
          
          -- pass through adaptive max pooling operation
--          local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[5])
--          local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
--          table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.indices:clone() })
        end
        
        -- process negative
        for i,x in ipairs(n) do
          local anchor = x[1]
          local l = anchor.layer
          local out = outputs[l]
          local delta_out = delta_outputs[l]
          local idx = anchor.index
          local v = out[idx]
          local d = delta_out[idx]
          
          cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 2)
          local dc = softmax:backward(v[{{1, 2}}], 2)
          d[{{1,2}}]:add(dc)

          if v[1] < v[2] then pbg_accu = pbg_accu+1; end

          -- pass through adaptive max pooling operation
--          local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[5])
--          local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
--          table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })
        end
        
-- Note by Bingbin: was used in the cnet part
--        if #roi_pool_state > 0 then
--          -- run backward pass over rois
--          for i,x in ipairs(roi_pool_state) do
--            amp.indices = x.indices
--            -- Note by Bingbin: pnet has delta_outputs[1:4]; 5 onwards are for cnet?
--            delta_outputs[5][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(cnet_input_planes, kh, kw)))
--          end
--        end
        
        -- backward pass of proposal network
        local gi = pnet:backward(img, delta_outputs)
        -- print(string.format('%f; pos: %d; neg: %d', gradient:max(), #p, #n))
        reg_count = reg_count + #p
        cls_count = cls_count + #p + #n
        pos_count = pos_count + #p
        neg_count = neg_count + #n
      end
      print('time per batch: ' .. (os.clock()-t_batch_start))
      
      -- scale gradient
      gradient:div(cls_count)
      
      local pcls = cls_loss / cls_count     -- proposal classification (bg/fg)
      local preg = reg_loss / reg_count     -- proposal bb regression
      if pos_count > 0 then
          pfg_accu = pfg_accu/pos_count
      else
          pfg_accu = -1
      end
      if neg_count > 0 then
          pbg_accu = pbg_accu/neg_count
      else
          pbg_accu = -1
      end
      local dcls, dreg = 0, 0
      
      print(string.format('prop: cls: %f / accu: fg=%f & bg=%f (%d), reg: %f (%d); det: cls: %f, reg: %f', 
        pcls, pfg_accu, pbg_accu, cls_count, preg, reg_count, dcls, dreg)
      )
      
      table.insert(stats.pcls, pcls)
      table.insert(stats.preg, preg)
      table.insert(stats.pfg_accu, pfg_accu)
      table.insert(stats.pbg_accu, pbg_accu)
     
      local loss = pcls + preg
      return loss, gradient
    end
    
    return lossAndGradient
end



