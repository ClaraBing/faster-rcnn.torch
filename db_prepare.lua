-- Modified
-- (Relatively) original version in joint/
require 'image'
require 'utilities'
require 'Anchors'

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

-- function processImage(cfg, norm_layer, img, rois)
-- fn: from training_data.traininig_set / validation_list / background_files
-- local rois = deep_copy(self.ground_truth[fn].rois)
-- local status, img = pcall(function () return load_image(fn, cfg.color_space,  cfg.examples_base_path) end)
-- Required:
-- training_data.ground_truth
--
function processImage(cfg, ground_truth)
  local aug = cfg.augmentation
  local norm_layer
  if cfg.normalization.method == 'contrastive' then
      norm_layer = nn.SpatialContrastiveNormalization(1, image.gaussian1D(cfg.normalization.width))
  else
      norm_layer = nn.Identity()
  end
 
  local img_roi_db = {}
  for fn, entry in pairs(ground_truth) do
    -- these two lines are to prepare 
    print('fn = ' .. fn)
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.example_base_path) end)
    if not status then
        print('Error loading image ' .. fn)
        goto continue
    end
    local rois = deep_copy(entry.rois)

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
    
    print(string.format('img size: %d * %d * %d', img:size()[1], img:size()[2], img:size()[3]))
    print(string.format('img type: %s', type(img[{{1}}])))
    img[1] = norm_layer:forward(img[{{1}}])   -- normalize luminance channel img
    img_roi_db[fn] = {image=deep_copy(img), rois=deep_copy(rois)}
    ::continue::
  end
  
  return img_roi_db
end

local cfg = dofile('config/imagenet_test.lua')
local training_data = load_obj('data_mine/ILSVRC2015_VID_test.t7')
save_obj('data_mine/img_roi_db.t7', processImage(cfg, training_data.ground_truth))
  
