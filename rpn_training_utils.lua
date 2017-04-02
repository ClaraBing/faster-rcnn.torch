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
require 'objective_rpn'

print('require set.')

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
    gnuplot.axis({ 0, #stats.pcls, 0, 6 }) 
  end; end

  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  
  gnuplot.plotflush()
end

function load_model(cfg, model_path, network_filename, cuda)

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)

  print('model:')
  print(model)

  print('model (printed by rpn.lua)')
  for i,block in ipairs(model.pnet.modules) do
    print('pnet module #' .. i)
    print(block)
  end
  print('\n')
--  for i,block in ipairs(model.cnet.modules) do
--    print('cnet module #' .. i)
--    print(block)
--  end
--  print('\n')
  
  if cuda then
--    model.cnet:cuda()
    model.pnet:cuda()
  end
  
  -- combine parameters from pnet and cnet into flat tensors
  -- local weights, gradient = model.pnet:getParameters() -- combine_and_flatten_parameters(model)
  local weights, gradient = combine_and_flatten_parameters(model.pnet)
  local training_stats
  if network_filename and #network_filename > 0 then
    local stored = load_obj(network_filename)
    training_stats = stored.stats
    weights:copy(stored.weights)
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

