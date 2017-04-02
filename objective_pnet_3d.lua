require 'cunn'
require 'BatchIterator_3d'
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

function create_objective_pnet(model, weights, gradient, batch_iterator, stats)
  local cfg = model.cfg
  local input_net = model.input_net
  local pnet = model.pnet

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
    local i = 1
    while i <= #examples do
      local anchor = examples[i][1]
      local fmSize = outputs[anchor.layer]:size()
      print('cleanAnchors: fmSize:')
      print(fmSize)
      -- Note by Bingbin: remove across-boundary anchors
      if anchor.index[2] > fmSize[2] or anchor.index[3] > fmSize[3] then
        print('removed anchor size: ' .. anchor.index[2] .. ' / ' .. anchor.index[3])
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
   
      -- enable dropouts 
      input_net:training()
      pnet:training()
      -- cnet:training()
      
      -- t_batch_start = os.clock()
      local batch = batch_iterator:nextTraining()
      print('\n\nnextTraining: #batch = ' .. #batch)
      local debug_out = true
      for i,x in ipairs(batch) do
        local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
        local p = x.positive        -- get positive and negative anchors examples
        local n = x.negative
        -- debug
        print('#p = ' .. #p .. ' / #n = ' .. #n)

        -- run forward convolution
        local pooled = input_net:forward(img)
        local fm_size = pooled:size()
        -- results of reshape: e.g. fm_t1_1, fm_t2_1, fm_t3_1, fm_t1_2, fm_t2_2, fm_t3_2, etc
        pooled = torch.reshape(pooled, fm_size[1]*fm_size[2], fm_size[3], fm_size[4])

        local outputs = pnet:forward(pooled)

        if (debug_out and i<3) then
          print('fm_size:')
          print(fm_size)
          print('fm_size after reshaping:')
          print(pooled:size())
          print('img size:')
          print(img:size())
          print('outputs:')
          print(outputs)
        end
        
        -- ensure all example anchors lie withing existing feature planes 
        print('before cleanAnchors: #p = ' .. #p .. ' / #n = ' .. #n)
        cleanAnchors(p, outputs)
        cleanAnchors(n, outputs)
        print('after cleanAnchors: #p = ' .. #p .. ' / #n = ' .. #n)
        
        -- clear delta values for each new image
        for j,out in ipairs(outputs) do
          if not delta_outputs[j] then
            delta_outputs[j] = torch.FloatTensor():cuda()
          end
          delta_outputs[j]:resizeAs(out)
          delta_outputs[j]:zero()
        end
        
        local roi_pool_state = {}
        local input_size = img:size()
        -- local cnetgrad
       
        -- process positive set
        for j,x in ipairs(p) do
          local anchor = x[1]
          local roi = x[2]
          local l = anchor.layer
          
          local out = outputs[l]
          local delta_out = delta_outputs[l]
           
          local idx = anchor.index
          local v = out[idx]
          local d = delta_out[idx]
            
          -- classification
          cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
          local dc = softmax:backward(v[{{1, 2}}], 1)

          if (debug_out and j<5 and i<3) then
            print('anchor-layer = ' .. l)
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
          
          if (debug_out and j<5 and i<3) then
              print('\nreg_out:')
              print(reg_out)
              print('\nreg_target:')
              print(reg_target)
              print('\ncls_out:')
              print(v)
          end

          reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * 10
          local dr = smoothL1:backward(reg_out, reg_target) * 10
          d[{{3,6}}]:add(dr)
          
          -- pass through adaptive max pooling operation
--          local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[5])
--          local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
--          table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.indices:clone() })
        end
        
        -- process negative
        for j,x in ipairs(n) do
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
        
       
        -- backward pass of proposal network
        local gi = pnet:backward(img, delta_outputs)
        -- Modified on April 2nd: for 3d
        local gi_size = gi:size()
        gi = torch.reshape(gi, gi_size[1]/3, 3, gi_size[2], gi_size[3])
        local input_gi = input_net:backward(img, gi)
        -- print(string.format('%f; pos: %d; neg: %d', gradient:max(), #p, #n))
        reg_count = reg_count + #p
        cls_count = cls_count + #p + #n
        pos_count = pos_count + #p
        neg_count = neg_count + #n
      end
      -- print('time per batch: ' .. (os.clock()-t_batch_start))
      
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
      
      -- print(string.format('prop: cls: %f / accu: fg=%f & bg=%f (%d), reg: %f (%d); det: cls: %f, reg: %f', 
      --  pcls, pfg_accu, pbg_accu, cls_count, preg, reg_count, dcls, dreg)
      -- )
      
      table.insert(stats.pcls, pcls)
      table.insert(stats.preg, preg)
      table.insert(stats.pfg_accu, pfg_accu)
      table.insert(stats.pbg_accu, pbg_accu)
     
      local loss = pcls + preg
      return loss, gradient
    end
    
    return lossAndGradient
end



