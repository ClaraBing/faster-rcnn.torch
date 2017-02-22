require 'image'
require 'utilities'
require 'Anchors_3d'

local debug = false

local BatchIterator_3d = torch.class('BatchIterator_3d')

local function randomize_order(...)
  local sets = { ... }
  for i,x in ipairs(sets) do
    if x.list and #x.list > 0 then   -- e.g. background examples are optional and randperm does not like 0 count
       -- Modified on Feb 19th: shuffle since only mid file presented
       x.order:randperm(#x.list)   -- shuffle
       -- x.order:range(1,#x.list)
    end
    x.i = 1   -- reset index positions
  end
end

local function next_entry(set)
  if set.i > #set.list then
    randomize_order(set)
  end
  
    -- Modified on Feb 12th: do not randomize
  local idx = set.order[set.i]
  local fn = set.list[idx]
  -- print(fn)
  local fidx = tonumber(fn:split('/')[9]:sub(1,6))
  local prev_fidx, next_fidx = fidx-3, fidx+3
  local fn_prefix = fn:sub(1, -11)
  local prev_fn = fn_prefix .. string.format('%06d', fidx-3) .. '.xml'
  local next_fn = fn_prefix .. string.format('%06d', fidx+3) .. '.xml'
  -- local pos = #(fn:split('/')) - 1
  -- local prev_fn, next_fn
  if io.open(prev_fn, 'r') == nil then
    prev_fn = fn
  end
  if io.open(next_fn, 'r') == nil then
    next_fn = fn
  end

  set.i = set.i + 1
  -- debug
  -- print('next_entry: ' .. prev_fn .. ' / ' .. fn .. ' / ' .. next_fn)
  return {[1]=prev_fn, [2]=fn, [3]=next_fn}
--  return prev_fn, fn, next_fn
end

local function transform_example(img, rois, fimg, froi)
  local result = {}
  local d = img:size()
  assert(d:size() == 3)
  img = fimg(img, d[3], d[2])   -- transform image
  local dn = img:size()
  local img_rect = Rect.new(0, 0, dn[3], dn[2])
  if rois then
    for i=1,#rois do
      local roi = rois[i]
      roi.rect = froi(roi.rect, d[3], d[2])   -- transform roi
      if roi.rect then
        roi.rect = roi.rect:clip(img_rect) 
        if not roi.rect:isEmpty() then
          result[#result+1] = roi
        end
      end 
    end
  end
  return img, result
end

local function scale(img, rois, scaleX, scaleY)
  scaleY = scaleY or scaleX
  return transform_example(img, rois, 
    function(img, w, h) return image.scale(img, math.max(1, w * scaleX), math.max(1, h * scaleY)) end,
    function(r, w, h) return r:scale(scaleX, scaleY) end
  )
end

local function hflip(img, rois)
  return transform_example(img, rois,
    function(img, w, h) return image.hflip(img) end,
    function(r, w, h) return Rect.new(w - r.maxX, r.minY, w - r.minX, r.maxY) end  
  )
end

local function vflip(img, rois)
  return transform_example(img, rois,
    function(img, w, h) return image.vflip(img) end,
    function(r, w, h) return Rect.new(r.minX, h - r.maxY, r.maxX, h - r.minY) end
  )
end

local function crop(img, rois, rect)
  return transform_example(img, rois,
    function(img, w, h) return image.crop(img, rect.minX, rect.minY, rect.maxX, rect.maxY) end,
    function(r, w, h) return r:clip(rect):offset(-rect.minX, -rect.minY) end 
  )
end

function BatchIterator_3d:__init(model, training_data)
  local cfg = model.cfg
  
  -- bounding box data (defined in pixels on original image)
  self.ground_truth = training_data.ground_truth 
  self.cfg = cfg
  
  if cfg.normalization.method == 'contrastive' then
    self.normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(cfg.normalization.width))
  else
    self.normalization = nn.Identity()
  end
  
  self.anchors = Anchors_3d.new(model.input_net, model.pnet, cfg.scales)
  
  -- index tensors define evaluation order
  self.training = { order = torch.IntTensor(), list = training_data.training_set }
  self.validation = { order = torch.IntTensor(), list = training_data.validation_set }
  self.background = { order = torch.IntTensor(), list = training_data.background_files or {} }
  
  randomize_order(self.training, self.validation, self.background)
end
  
function BatchIterator_3d:processImage(img, rois)
  local cfg = self.cfg
  local aug = cfg.augmentation
  
  -- determine optimal resize
  local img_size = img:size()
  local tw, th = find_target_size(img_size[3], img_size[2], cfg.target_smaller_side, cfg.max_pixel_size)
  
  local scale_X, scale_Y = tw / img_size[3], th / img_size[2]

  -- random scaling
  if aug.random_scaling and aug.random_scaling > 0 then
    scale_X = tw * (math.random() - 0.5) * aug.random_scaling / img_size[3] 
    scale_Y = scale_X + (math.random() - 0.5) * aug.aspect_jitter
  end
  
  img, rois = scale(img, rois, scale_X, scale_Y)
  
  -- crop image to final size if we upsampled at least one dimension
  img_size = img:size()
  if img_size[3] > tw or img_size[2] > th then
    tw, th = math.min(tw, img_size[3]), math.min(th, img_size[2])
    local crop_rect = Rect.fromXYWidthHeight(
      math.floor(math.random() * (img_size[3]-tw)), 
      math.floor(math.random() * (img_size[2]-th)), 
      tw, 
      th
    )
    img, rois = crop(img, rois, crop_rect)
  end
  
  -- horizontal flip operation
  if aug.hflip and aug.hflip > 0 then 
    if math.random() < aug.hflip then
      img, rois = hflip(img, rois)
    end
  end
  
  -- vertical flip operation
  if aug.vflip and aug.vflip > 0 then
    if math.random() < aug.vflip then
      img, rois = vflip(img, rois)
    end
  end
  
  if cfg.normalization.centering then
    for i = 1,3 do
      img[i] = img[i]:add(-img[i]:mean())
    end
  end
  
  if cfg.normalization.scaling then
  for i = 1,3 do
      local s = img[i]:std()
      if s > 1e-8 then
        img[i] = img[i]:div(s)
      end
    end
  end
  
  img[1] = self.normalization:forward(img[{{1}}])   -- normalize luminance channel img
  
  return img, rois
end
  
function BatchIterator_3d:nextTraining(count)
  local cfg = self.cfg
  local batch = {}
  count = count or cfg.batch_size
  
  -- use local function to allow early exits in case of to image load failures
  local function try_add_next()
    -- Modified on Feb 12th: add prev_fn & next_fn
    local fn = next_entry(self.training)
--    local rois = deep_copy(self.ground_truth[fn[2]].rois)   -- copy RoIs ground-truth data (will be manipulated)
  
    -- Modified on Feb 12th: change to 3d
    img_3d = {}
    rois_3d = {}
    for i=1,3 do
      -- load image, wrap with pcall since image net contains invalid non-jpeg files
      local status, img = pcall(function () return load_image(fn[i], cfg.color_space, cfg.examples_base_path) end)
      if not status then
        -- pcall failed, corrupted image file?
        print(string.format("Invalid image '%s': %s", fn[i], img))
        return 0
      end
  
      local img_size = img:size()
      if img:nDimension() ~= 3 or img_size[1] ~= 3 then
        print(string.format("Warning: Skipping image '%s'. Unexpected channel count: %d (dim: %d)", fn, img_size[1], img:nDimension()))
        return 0
      end 
      
      local rois = deep_copy(self.ground_truth[fn[i]].rois)
      local img, rois = self:processImage(img, rois)
      img_size = img:size()        -- get final size
      if img_size[2] < 128 or img_size[3] < 128 then
          -- notify user about skipped image
          print(string.format("Warning: Skipping image '%s'. Invalid size after process: (%dx%d)", fn, img_size[3], img_size[2]))  
        return 0
      end

      -- Modified on Feb 17th: change deep_copy to this method
      local tmp_img = torch.Tensor(img:size()):zero()
      tmp_img[1]:add(img[1])
      tmp_img[2]:add(img[2])
      tmp_img[3]:add(img[3])
      img_3d[i] = tmp_img

      rois_3d[i] = deep_copy(rois)
    end
    
    -- Modified on Feb 12th: use the current image (i.e. index=2)
    local img_size = img_3d[2]:size()
    local rois = rois_3d[2]
    -- end of Modified
    -- find positive examples
    local img_rect = Rect.new(0, 0, img_size[3], img_size[2])
    local positive = self.anchors:findPositive(rois, img_rect, cfg.positive_threshold, cfg.negative_threshold, cfg.best_match)
    -- debug
    if debug then
      print('img_size (nextTraining):')
      print(img_size)
      print('positive[1] (after findPositive):')
      print(positive[1])
    end
    
    -- random negative examples
    local negative = self.anchors:sampleNegative(img_rect, rois, cfg.negative_threshold, 16)
    local count = #positive + #negative
     
    if cfg.nearby_aversion then
      local nearby_negative = {}
      -- add all nearby negative anchors
      for i,p in ipairs(positive) do
        local cx, cy = p[1]:center()
        local nearbyAnchors = self.anchors:findNearby(cx, cy)
        for i,a in ipairs(nearbyAnchors) do
          if Rect.IoU(p[1], a) < cfg.negative_threshold then
            table.insert(nearby_negative, { a })
          end
        end
      end
      
      local c = math.min(#positive, count)
      shuffle_n(nearby_negative, c)
      for i=1,c do
        table.insert(negative, nearby_negative[i])
        count = count + 1
      end
    end
    
    -- debug boxes
    if true then
      local dimg = image.yuv2rgb(img_3d[2])
      local gray_img = torch.Tensor(img_size[3], img_size[2]):zero()
      gray_img:add(0.21, dimg[1]):add(0.72, dimg[2]):add(0.07, dimg[3])
      dimg[1] = dimg[1]:zero():add(gray_img)
      dimg[2] = dimg[2]:zero():add(gray_img)
      dimg[3] = dimg[3]:zero():add(gray_img)

      local red = torch.Tensor({1,0,0})
      local green = torch.Tensor({0,1,0})
      local blue = torch.Tensor({0,0,1})
      -- local white = torch.Tensor({1,1,1})
      
      for i=1,#negative do
        draw_rectangle_gray(dimg, negative[i][1], red)
      end
      for i=1,#positive do
        draw_rectangle_gray(dimg, positive[i][1], green)
      end
      for i=1,#rois do
        draw_rectangle_gray(dimg, rois[i].rect, blue)
      end
      image.saveJPG(string.format('img_out_mine/anchor/test-anchors%d_gray.jpg', self.training.i), dimg)
    end

    -- Modified on Feb 12th: change img to 3D: nInputPlane x time x W x H
    -- img_3d = torch.cat({img_3d[1], img_3d[2], img_3d[3]})
    if debug then
      print('img_3d size: ')
      print(img_3d[1]:size())
      print('concated img_3d size:')
      print(torch.cat({img_3d[1], img_3d[2], img_3d[3]}, 1):size())
      print('positive[1]:')
      print(positive[1])
      print('negative[1]:')
      print(negative[1])
      print(string.format("'%s' (%dx%d); p: %d; n: %d", fn, img_size[2], img_size[3], #positive, #negative))
    end
    img_3d = torch.reshape(torch.cat({img_3d[1], img_3d[2], img_3d[3]}, 1), img_size[1], 3, img_size[2], img_size[3])
    table.insert(batch, { img = img_3d, positive = positive, negative = negative })
    return count
  end -- end of try_add_next
  
  -- add a background examples
  if #self.background.list > 0 then
    -- Modified on Feb 12th: add WARNING when using bg img
    print('WARNING: using self.background.list (in BatchIterator_3d:nextTraining)')
    local fn = next_entry(self.background)
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.background_base_path) end)
    if status then
      img = self:processImage(img)
      local img_size = img:size()        -- get final size
      if img_size[2] >= 128 and img_size[3] >= 128 then
        local img_rect = Rect.new(0, 0, img_size[3], img_size[2])
        local negative = self.anchors:sampleNegative(img_rect, {}, 0, math.floor(count * 0.05))   -- add 5% negative samples per batch
        table.insert(batch, { img = img, positive = {}, negative = negative })
        count = count - #negative
        print(string.format('background: %s (%dx%d)', fn, img_size[3], img_size[2]))
      end
    else
      -- pcall failed, corrupted image file?
      print(string.format("Invalid image '%s': %s", fn, img))
    end
  end
  
  while count > 0 do
    count = count - try_add_next()
  end
  
  return batch
end

function BatchIterator_3d:nextValidation(count)
  local cfg = self.cfg
  local batch = {}
  count = count or 1
  
  -- use local function to allow early exits in case of to image load failures
  while count > 0 do
    local prev_fn, fn, next_fn = next_entry(self.validation)
  
    -- load image, wrap with pcall since image net contains invalid non-jpeg files
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.examples_base_path) end)
    if not status then
      -- pcall failed, corrupted image file?
      print(string.format("Invalid image '%s': %s", fn, img))
      goto continue
    end

    local img_size = img:size()
    if img:nDimension() ~= 3 or img_size[1] ~= 3 then
      print(string.format("Warning: Skipping image '%s'. Unexpected channel count: %d (dim: %d)", fn, img_size[1], img:nDimension()))
      goto continue
    end 
    
    local rois = deep_copy(self.ground_truth[fn].rois)   -- copy RoIs ground-truth data (will be manipulated, e.g. scaled)
    local img, rois = self:processImage(img, rois)
    img_size = img:size()        -- get final size
    if img_size[2] < 128 or img_size[3] < 128 then
      print(string.format("Warning: Skipping image '%s'. Invalid size after process: (%dx%d)", fn, img_size[3], img_size[2]))  
      goto continue
    end
      
    table.insert(batch, { img = img, rois = rois })
  
    count = count - 1
    ::continue::
  end
  
  return batch  
end
