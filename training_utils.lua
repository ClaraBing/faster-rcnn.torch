require 'torch'
require 'pl'
require 'optim'
require 'image'
require 'nngraph'
require 'cunn'
require 'nms'
require 'gnuplot'

require 'utilities'
require 'Anchors'
require 'BatchIterator'
require 'objective_pnet'

print('training_utils: require set.')

function plot_training_progress(prefix, varname, stats)
  local fn = prefix .. '_' .. varname .. '_progress.png'
  gnuplot.pngfigure(fn)
  gnuplot.title('Traning progress over time')

  local xs = torch.range(1, #stats.pcls)

  if varname == 'paccu' then
    gnuplot.plot{
      {'pfg_accu', xs, torch.Tensor(stats.pfg_accu), '-'},
      {'pbg_accu', xs, torch.Tensor(stats.pbg_accu), '-'},
    }
    gnuplot.axis({ 0, #stats.pcls, 0, 1 }) 
  else if varname == 'daccu' then
      gnuplot.plot{
        {'dfg_accu', xs, torch.Tensor(stats.dfg_accu), '-'},
        {'dbg_accu', xs, torch.Tensor(stats.dbg_accu), '-'},
      }
      gnuplot.axis({ 0, #stats.pcls, 0, 1 }) 
  else
    local var
    if varname == 'pcls' then
      var = stats.pcls
    end
    if varname == 'preg' then
      var = stats.preg
    end
    if varname == 'dcls' then
      var = stats.dcls
    end
    if varname == 'dreg' then
      var = stats.dreg
    end
  
    gnuplot.plot{
      {varname, xs, torch.Tensor(var), '-'}
    }
    gnuplot.axis({ 0, #stats.pcls, 0, 2 }) 
  end; end

  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  
  gnuplot.plotflush()
end

function load_model(cfg, model_path, network_filename, cuda)

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)

  print('model (printed by load_model)')
  for i,block in ipairs(model.pnet.modules) do
    print('pnet module #' .. i)
    print(block)
  end
  print('\n')
  for i,block in ipairs(model.cnet.modules) do
    print('cnet module #' .. i)
    print(block)
  end
  print('\n')
  
  if cuda then
    model.cnet:cuda()
    model.pnet:cuda()
  end
  
  local training_stats
  local weights, gradient
  if network_filename and #network_filename ~= 0 then
      print('load_model: restored')
      -- restore from t7 model
      local stored = load_obj(network_filename)
      training_stats = stored.stats
      weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet)
      print('Weight size:')
      print(weights:size())
      print('Stoed weight size:')
      print(stored.weights:size())
      weights:copy(stored.weights)
  elseif cfg.pretrain_prototxt and cfg.pretrain_model and #cfg.pretrain_model ~= '' then
      print('load_model: use Caffe model')
      -- new training - get weights from pretrained models
      require 'loadcaffe'
      local pretrained_model = loadcaffe.load(cfg.pretrain_prototxt, cfg.pretrain_model)
      local w, g = pretrained_model:parameters()
      training_stats = nil

      -- combine parameters from pnet and cnet into flat tensors
      weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet, w, g)
      -- debug
      print('load_model: weights:size():')
      print(weights:size())
  else
    weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet)
    end
  return model, weights, gradient, training_stats
end

function load_model_3d(cfg, model_path, network_filename, cuda)

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)

  print('model (printed by load_model)')
  for i,block in ipairs(model.pnet.modules) do
    print('pnet module #' .. i)
    print(block)
  end
  print('\n')
  for i,block in ipairs(model.cnet.modules) do
    print('cnet module #' .. i)
    print(block)
  end
  print('\n')
  
  if cuda then
    model.input_net:cuda()
    model.cnet:cuda()
    model.pnet:cuda()
  end
  
  local training_stats
  local weights, gradient
  if network_filename and #network_filename ~= 0 then
      print('load_model: restored')
      -- restore from t7 model
      local stored = load_obj(network_filename)
      training_stats = stored.stats
      weights, gradient = combine_and_flatten_parameters_3d(model.input_net, model.pnet, model.cnet)
      print('Weight size:')
      print(weights:size())
      print('Stoed weight size:')
      print(stored.weights:size())
      weights:copy(stored.weights)
  elseif cfg.pretrain_prototxt and cfg.pretrain_model and #cfg.pretrain_model ~= '' then
      print('load_model: use Caffe model')
      -- new training - get weights from pretrained models
      require 'loadcaffe'
      local pretrained_model = loadcaffe.load(cfg.pretrain_prototxt, cfg.pretrain_model)
      local w, g = pretrained_model:parameters()
      training_stats = nil

      -- combine parameters from pnet and cnet into flat tensors
      weights, gradient = combine_and_flatten_parameters_3d(model.input_net, model.pnet, model.cnet, w, g)
      -- debug
      print('load_model: weights:size():')
      print(weights:size())
  else
      print('load_model: no restore/pretrain; train from scratch.')
      weights, gradient = combine_and_flatten_parameters_3d(model.input_net, model.pnet, model.cnet)
  end

  return model, weights, gradient, training_stats
end

function load_image_auto_size(fn, target_smaller_side, max_pixel_size, color_space)
  local img = image.load(path.join(base_path, fn), 3, 'float')
  local dim = img:size()
  
  local w, h
  if dim[2] < dim[3] then
    -- height is smaller than width, set h to target_size
    w = math.min(dim[3] * target_smaller_side/dim[2], max_pixel_size)
    h = dim[2] * w/dim[3]
  else
    -- width is smaller than height, set w to target_size
    h = math.min(dim[2] * target_smaller_side/dim[1], max_pixel_size)
    w = dim[3] * h/dim[2]
  end
  
  img = image.scale(img, w, h)
  
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end

  return img, dim
end

