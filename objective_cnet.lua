-- Same as the original one except for debugging prints
--

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

function create_objective_cnet(model, outputs, weights, gradient, batch_iterator, stats)
  local cfg = model.cfg
  local pnet = model.pnet
  local cnet = model.cnet
  
  local bgclass = cfg.class_count + 1   -- background class
  local anchors = batch_iterator.anchors    
  local localizer = Localizer.new(pnet.outnode.children[5])
    
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cnll = nn.ClassNLLCriterion():cuda()
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
      -- Note by Bingbin: remove across-boundary anchors
      if anchor.index[2] > fmSize[2] or anchor.index[3] > fmSize[3] then
        print('cleanAnchors: exceed -> crop')
        print(anchor.index)
        print(fmSize)
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

      local delta_outputs = {}
   
      -- statistics for fine-tuning and classification stage
      local creg_loss, creg_count = 0, 0
      local ccls_loss, ccls_count = 0, 0
      
      -- enable dropouts 
      pnet:training()
      cnet:training()
      
      t_batch_start = os.clock()
      local batch = batch_iterator:nextTraining()
      print('Begin of batch (len = ' .. #batch .. ')')
      for i,x in ipairs(batch) do
        local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
        local p = x.positive        -- get positive and negative anchors examples
        local n = x.negative

        -- debug
        print('Img fn = ' .. x.fn)
        print('img size: ' .. img:size()[1] .. ' * ' .. img:size()[2] .. ' * ' .. img:size()[3])

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
        local cnetgrad

        -- positive examples
        local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[5])
        local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
        table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.    indices:clone() })
       
         -- fine-tuning STAGE
         -- pass extracted roi-data through classification network
        
        -- create cnet input batch
        if #roi_pool_state > 0 then
          local cinput = torch.CudaTensor(#roi_pool_state, kh * kw * cnet_input_planes)
          local cctarget = torch.CudaTensor(#roi_pool_state)
          local crtarget = torch.CudaTensor(#roi_pool_state, 4):zero()
          
          for i,x in ipairs(roi_pool_state) do
            cinput[i] = x.output
            if x.roi then
              -- positive example
              cctarget[i] = x.roi.class_index
              crtarget[i] = Anchors.inputToAnchor(x.reg_proposal, x.roi.rect)   -- base fine tuning on proposal
            else
              -- negative example
              cctarget[i] = bgclass
            end
          end
          
          -- process classification batch 
          local coutputs = cnet:forward(cinput)
          
          -- compute classification and regression error and run backward pass
          local crout = coutputs[1]
          --print(crout)
          
          crout[{{#p + 1, #roi_pool_state}, {}}]:zero() -- ignore negative examples
          creg_loss = creg_loss + smoothL1:forward(crout, crtarget) * 10
          local crdelta = smoothL1:backward(crout, crtarget) * 10
          
          local ccout = coutputs[2]  -- log softmax classification
          local curr_loss = cnll:forward(ccout, cctarget)
          ccls_loss = ccls_loss + curr_loss 
          local ccdelta = cnll:backward(ccout, cctarget)
          
          local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })
          
          -- run backward pass over rois
          for i,x in ipairs(roi_pool_state) do
            amp.indices = x.indices
            delta_outputs[5][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(cnet_input_planes, kh, kw)))
          end
        end
        
        creg_count = creg_count + #p
        ccls_count = ccls_count + 1
        
      end
      print('time per batch: ' .. (os.clock()-t_batch_start))
      
      -- scale gradient
      gradient:div(cls_count)
      
      local pcls = 0     -- proposal classification (bg/fg)
      local preg = 0     -- proposal bb regression
      local dcls = ccls_loss / ccls_count   -- detection classification
      local dreg = creg_loss / creg_count   -- detection bb finetuning
      
      print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f, reg: %f', 
        pcls, cls_count, preg, reg_count, dcls, dreg)
      )
      
      table.insert(stats.pcls, pcls)
      table.insert(stats.preg, preg)
      table.insert(stats.dcls, dcls)
      table.insert(stats.dreg, dreg)
      
      local loss = pcls + preg
      return loss, gradient
    end
    
    return lossAndGradient
end
