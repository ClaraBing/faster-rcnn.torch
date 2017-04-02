require 'torch'
require 'pl'
require 'optim'
require 'image'
require 'nngraph'
require 'cunn'
require 'nms'
require 'gnuplot'

require 'training_utils'
require 'utilities'
require 'Anchors'
require 'BatchIterator'
require 'objective_pnet'
require 'Detector_2stage'

print('require set.')


-- command line options
cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
-- cmd:option('-cfg', 'config/imagenet.lua', 'configuration file')
cmd:option('-cfg', 'config/imagenet_test.lua', 'configuration file')
-- cmd:option('-model', 'models/vgg_small.lua', 'model factory file')
cmd:option('-model', 'models/new-zf.lua', 'model factory file')
-- cmd:option('-name', '/disk/bingbin/faster-rcnn-torch/imgnet', 'experiment name, snapshot prefix') 
-- cmd:option('-name', 'output_mine/pnet/imgnet', 'experiment name, snapshot prefix') 
cmd:option('-train', 'data_mine/ILSVRC2015_VID_test.t7', 'training data file name')
-- cmd:option('-train', 'data_mine/ILSVRC2015_VID_sampled_double.t7', 'training data file name')
-- Modified on Feb 4th: set restore model
cmd:option('-restore', '', 'network snapshot file name to load')
-- cmd:option('-restore', 'output_mine/orig_80k_batch128_lr5_per_2k/imgnet_030000.t7', 'network snapshot file name to load')
--cmd:option('-pnet', 'output_mine/pnet/anchor_wrong_layer_imgnet_005000.t7', 'pnet for generating proposals')
cmd:option('-pnet', 'output_mine/pnet/imgnet_008000.t7', 'pnet for generating proposals')
cmd:option('-snapshot', 1000, 'snapshot interval')
cmd:option('-plot', 100, 'plot training progress interval')
cmd:option('-lr', 1E-4, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'rmsprop', 'Optimizer')
-- Added on Mar 4th: add dataset to save rois
cmd:option('-dataset', 'output_mine/get_prop/dataset_conv5.obj', 'Dataset for storing rois obtained from pnet training')

cmd:text('=== Misc ===')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU)')
cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')

print('Command line args:')
local opt = cmd:parse(arg or {})
print(opt)

print('Options:')
local cfg = dofile(opt.cfg)
print(cfg)

-- system configuration
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid + 1)  -- nvidia tools start counting at 0
torch.setnumthreads(opt.threads)
if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end

function get_train_proposal(cfg, model_path, training_data_filename, network_filename, dataset_save_name)
  -- load trainnig data
  local t_start = os.clock()
  local training_data = load_obj(training_data_filename)
  -- training_data = training_data.training_set
  print(string.format('training_data loaded: %.3f\n', os.clock()-t_start))
  
  -- load model
  t_start = os.clock()
  local model = load_model(cfg, model_path, network_filename, true)
  print(string.format('model loaded: %.3f\n', os.clock()-t_start))

  local batch_iterator = BatchIterator.new(model, training_data, false) -- do not randomize
  print(string.format('batch_iterator created: %.3f\n', os.clock()-t_start))
    
  -- create detector
  local d = Detector_2stage.new(model)
  local dataset = {}    
  local img_idx_map = {}
  local curr_idx = 1
  print('#training_data = ' .. #training_data.training_set)
  for i=1,#training_data.training_set do
    t_start = os.clock()
    -- pick random validation image
    -- local b = batch_iterator:nextValidation(1)[1]
    local b = batch_iterator:getImage('training')[1]
    local img = b.img
    
    -- local candidates, outputs = d:detect(img, true)
    local cinputs = d:detect(img, true)
    -- break
    -- table.insert(dataset, {fn=b.fn, gt_rois=b.rois, rois=candidates, outputs=outputs}) -- proposal only
    local img_id = b.fn:sub(47, 104)
    -- if dataset[img_id] ~= nil then
    if img_idx_map[img_id] ~= nil and cinputs ~= nil then
      -- print('ERROR: file \'' .. b.fn  .. '\' encountered twice.\n')
      -- return
      for j,v in ipairs(cinputs) do
        table.insert(dataset[img_idx_map[img_id]].boxes, {[1]=v.r['minY'], [2]=v.r['minX'], [3]=v.r['maxY'], [4]=v.r['maxX']})
      end
    else
      -- dataset[b.fn] = {gt_rois=b.rois, rois=candidates}
      print('img_id: ' .. img_id)
      img_idx_map[img_id] = curr_idx
      local boxes = {}
      for j,v in ipairs(candidates) do
        -- [y1 x1 y2 x2]
        boxes[j] = {[1]=v.r['minY'], [2]=v.r['minX'], [3]=v.r['maxY'], [4]=v.r['maxX']}
      end
      dataset[curr_idx] = {images=img_id, boxes=boxes}
      curr_idx = curr_idx+1
    end
    print(string.format('img %d: %.3f\n', i, os.clock()-t_start))
     if i%10 == 0 then
       local images, boxes = {}, {}
       for j,v in ipairs(dataset) do
           images[j] = v.images
           boxes[j] = v.boxes
       end
       save_obj(dataset_save_name, {images=images, boxes=boxes})
     end
  end
   local images, boxes = {}, {}
   print('length: #dataset = ' .. #dataset)
   for j,v in ipairs(dataset) do
       images[j] = v.images
       boxes[j] = v.boxes
   end
   save_obj(dataset_save_name, {dataset=dataset, images=images, boxes=boxes})
   -- save_obj('roidb_' .. dataset_save_name, {images=images, boxes=boxes})
   save_obj('dataset_' .. dataset_save_name, dataset)
end

-- evaluation_demo(cfg, opt.model, opt.train, opt.restore)
get_train_proposal(cfg, opt.model, opt.train, opt.pnet, opt.dataset)
-- evaluation_demo(cfg, opt.model, opt.train, 'output_mine/imgnet.t7')


