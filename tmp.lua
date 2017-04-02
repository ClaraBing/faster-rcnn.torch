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
cmd:option('-model', 'models/zf_prelu.lua', 'model factory file')
-- cmd:option('-name', '/disk/bingbin/faster-rcnn-torch/imgnet', 'experiment name, snapshot prefix') 
cmd:option('-name', 'output_mine/pnet/tmp_imgnet', 'experiment name, snapshot prefix') 
cmd:option('-train', 'data_mine/ILSVRC2015_VID_test.t7', 'training data file name')
-- cmd:option('-train', 'data_mine/ILSVRC2015_VID_sampled_double.t7', 'training data file name')
cmd:option('-restore', 'output_mine/pnet/imgnet_066000.t7', 'network snapshot file name to load')
-- cmd:option('-restore', '', 'network snapshot file name to load')
-- cmd:option('-pnet', 'output_mine/pnet/imgnet_005000.t7', 'pnet for generating proposals')
cmd:option('-snapshot', 1000, 'snapshot interval')
cmd:option('-plot', 200, 'plot training progress interval')
cmd:option('-lr', 1E-4, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'rmsprop', 'Optimizer')
-- Added on Mar 4th: add dataset to save rois
-- cmd:option('-dataset', 'output_mine/tmp/dataset.obj', 'Dataset for storing rois obtained from pnet training')

cmd:text('=== Misc ===')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-gpuid', 1, 'device ID (CUDA), (use -1 for CPU)')
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

function pnet_training(cfg, model_path, snapshot_prefix, training_data_filename, network_filename)
  print('Reading training data file \'' .. training_data_filename .. '\'.')
  local training_data = load_obj(training_data_filename)
  local file_names = keys(training_data.ground_truth)
  print(string.format("Training data loaded. Dataset: '%s'; Total files: %d; classes: %d; Background: %d)", 
      training_data.dataset_name, 
      #file_names,
      #training_data.class_names,
      #training_data.background_files))
  
  -- create/load model
  local model, weights, gradient, training_stats = load_model(cfg, model_path, network_filename, true)
  if not training_stats then
    training_stats = { pcls={}, preg={}, dcls={}, dreg={}, pfg_accu={}, pbg_accu={} }
  end
  
  local batch_iterator = BatchIterator.new(model, training_data, true)
  local eval_objective_grad = create_objective_pnet(model, weights, gradient, batch_iterator, training_stats)
  
  local rmsprop_state = { learningRate = opt.lr, alpha = opt.rms_decay }
  --local nag_state = { learningRate = opt.lr, weightDecay = 0, momentum = opt.rms_decay }
  -- local sgd_state = { learningRate = opt.lr, weightDecay = 0.0005, momentum = 0.9 }
  
  local timer = os.clock()
  for i=1,100000 do
    if i % 10000 == 0 then
      opt.lr = opt.lr / 10
      rmsprop_state.lr = opt.lr
      print('opt.lr updated to ' .. opt.lr .. ' at iter #' .. i)
    end
  
    local _, loss = optim.rmsprop(eval_objective_grad, weights, rmsprop_state)
    -- local _, loss = optim.nag(eval_objective_grad, weights, nag_state)
    -- local _, loss = optim.sgd(eval_objective_grad, weights, sgd_state)
   
    if i%opt.plot == 0 then
      plot_training_progress(snapshot_prefix, 'pcls', training_stats)
      plot_training_progress(snapshot_prefix, 'preg', training_stats)
      plot_training_progress(snapshot_prefix, 'paccu', training_stats)
      -- plot_training_progress(snapshot_prefix, 'dcls', training_stats)
      -- plot_training_progress(snapshot_prefix, 'dreg', training_stats)

      local len = #training_stats.pcls
      local pcls, preg, pfg_accu, pbg_accu = 0,0,0,0
      for k=len-opt.plot+1, len do
          pcls = pcls + training_stats.pcls[k]
          preg = preg + training_stats.preg[k]
          pfg_accu = pfg_accu + training_stats.pfg_accu[k]
          pbg_accu = pbg_accu + training_stats.pbg_accu[k]
      end
      pcls = pcls / opt.plot
      preg = preg / opt.plot
      pfg_accu = pfg_accu / opt.plot
      pbg_accu = pbg_accu / opt.plot
      print('\n\n==== Iter ' .. i .. ' ====')
      print(string.format('avg time per batch: %f / loss: %f / lr: %f', (os.clock()-timer)/200, loss[1], opt.lr))
      timer = os.clock()
      print(string.format('stats: cls: %f / reg: %f / fg_accu: %f / bg_accu: %f', pcls, preg, pfg_accu, pbg_accu))
    end
    
    if i%opt.snapshot == 0 then
      -- save snapshot
      save_model(string.format('%s_%06d.t7', snapshot_prefix, i), weights, opt, training_stats)
    end
  end

  -- save trained model
  save_model(string.format('%s.t7', snapshot_prefix), weights, opt, training_stats)

  
  -- compute positive anchors, add anchors to ground-truth file
end

function evaluation_demo(cfg, model_path, training_data_filename, network_filename)
  -- load trainnig data
  local t_start = os.clock()
  local training_data = load_obj(training_data_filename)
  print(string.format('training_data loaded: %.3f\n', os.clock()-t_start))
  
  -- load model
  t_start = os.clock()
  local model = load_model(cfg, model_path, network_filename, true)
  print(string.format('model loaded: %.3f\n', os.clock()-t_start))

  t_start = os.clock()
  local batch_iterator = BatchIterator.new(model, training_data, true)
  print(string.format('batch_iterator created: %.3f\n', os.clock()-t_start))
    
  -- create detector
  local d = Detector_2stage(model)
    
  for i=1,20 do
  
    -- pick random validation image
    -- local b = batch_iterator:nextValidation(1)[1]
    local b = batch_iterator:nextTraining(1)[1]
    local img = b.img
    
    t_start = os.clock()
    local matches = d:detect(img)
    print(string.format('img %d: %.3f\n', i, os.clock()-t_start))
    -- print('distinct')
    -- print(#matches)

    img = image.yuv2rgb(img)
    -- draw bounding boxes and save image
    for i,m in ipairs(matches) do
      -- Modified on Jan 23rd
--      local keys = {}
--      for k,v in pairs(match) do
--        table.insert(keys, k)
--      end
      print('m:\n')
      print(m)
      print('\n\n\n')
      draw_rectangle(img, m.r, green)
    end
    -- Modified on Jan 23rd
    if #matches ~= 0 then
      print('have matches. Break.')
      image.saveJPG(string.format('output%d.jpg', i), img)
      break
    end
    -- image.saveJPG(string.format('output%d.jpg', i), img)
  end
  
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
  print('#training_data = ' .. #training_data.training_set)
  for i=1,#training_data.training_set do
    t_start = os.clock()
    -- pick random validation image
    -- local b = batch_iterator:nextValidation(1)[1]
    local b = batch_iterator:getImage('training')[1]
    local img = b.img
    
    local candidates, outputs = d:detect(img, true)
    -- table.insert(dataset, {fn=b.fn, gt_rois=b.rois, rois=candidates, outputs=outputs}) -- proposal only
    if dataset[b.fn] ~= nil then
      -- print('ERROR: file \'' .. b.fn  .. '\' encountered twice.\n')
      -- return
      table.insert(dataset[b.fn].gt_rois, b.rois)
      table.insert(dataset[b.fn].rois, candidates)
    else
      dataset[b.fn] = {gt_rois=b.rois, rois=candidates}
    end
    print(string.format('img %d: %.3f\n', i, os.clock()-t_start))
  end
  save_obj(dataset_save_name, dataset)
end

pnet_training(cfg, opt.model, opt.name, opt.train, opt.restore)
-- evaluation_demo(cfg, opt.model, opt.train, opt.restore)
-- get_train_proposal(cfg, opt.model, opt.train, opt.pnet, opt.dataset)
-- evaluation_demo(cfg, opt.model, opt.train, 'output_mine/imgnet.t7')


