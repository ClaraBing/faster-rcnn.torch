require 'Rect'
require 'utilities'

local Localizer_3d = torch.class('Localizer_3d')

function Localizer_3d:__init(inet_outnode, pnet_outnode)

  local function trace_modules(node)
    print('tracing modules') --debug
    local modules = {}
    local function add_modules(c)
      if c.modules then
        for i=#c.modules,1,-1 do
          add_modules(c.modules[i])
          print(c.modules[i]) -- debug
        end
      else
        table.insert(modules, c)
      end
    end  
    while node do
      if node.data.module then
        add_modules(node.data.module)
      end
      node = node.children and node.children[1]
    end
    return reverse(modules)
  end -- trace_modules
  
  local function create_layer_info(modules, do_print)
    local info = {}
    for i,m in ipairs(modules) do
      if m.kW and m.kH then
        table.insert(info, { kW=m.kW, kH=m.kH, dW=m.dW or 1, dH=m.dH or 1, padW=m.padW or 0, padH=m.padH or 0 })
      end
    end
    if do_print then
      print('self.layers:')
      for i, layer in ipairs(info) do
        local info_str = string.format('dH=%d / dW=%d / kH=%d / kW=%d / padH=%d / padW=%d', info[i].dH, info[i].dW, info[i].kH, info[i].kW, info[i].padH, info[i].padW)
        print('#' .. i .. ': ' .. info_str)
      end
    end
    -- print(info)
    return info
  end -- create_layer_info
  
  print('inet layers')
  local inet_layers = create_layer_info(trace_modules(inet_outnode), false)
  print('(type = )' .. type(inet_layers) .. '\n\n')
  print('pnet layers')
  local pnet_layers = create_layer_info(trace_modules(pnet_outnode), true)
  self.layers = {}
  self.layers[1] = inet_layers[1]
  self.layers[2] = inet_layers[2]
  for i=2,#pnet_layers do
    self.layers[1+i] = pnet_layers[i]
  end
  print('self.layers(final):')
  for i, layer in ipairs(self.layers) do
    local info_str = string.format('dH=%d / dW=%d / kH=%d / kW=%d / padH=%d / padW=%d', layer.dH, layer.dW, layer.kH, layer.kW, layer.padH, layer.padW)
    print('#' .. i .. ': ' .. info_str)
  end
  print('### end of self.layers\n\n')
  -- self.layers = create_layer_info(trace_modules(pnet_outnode))
end

function Localizer_3d:inputToFeatureRect(rect, layer_index)
  layer_index = layer_index or #self.layers
  -- print('inputToFeatureRect: layer_index = ' .. layer_index) -- debug
  for i=1,layer_index do
    local l = self.layers[i]
    if l.dW < l.kW then
      rect = rect:inflate((l.kW-l.dW), (l.kH-l.dH))
    end

    rect = rect:offset(l.padW, l.padH)
    
    -- reduce size, keep only filters that fit completely into the rect (valid convolution)
    rect.minX = rect.minX / l.dH
    rect.minY = rect.minY / l.dH
    if (rect.maxX-l.kW) % l.dW == 0 then
      rect.maxX = math.max((rect.maxX-l.kW)/l.dW + 1, rect.minX+1)
    else
      rect.maxX = math.max(math.ceil((rect.maxX-l.kW) / l.dW) + 1, rect.minX+1)
    end
    if (rect.maxY-l.kH) % l.dH == 0 then
      rect.maxY = math.max((rect.maxY-l.kH)/l.dW + 1, rect.minY+1)
    else
      rect.maxY = math.max(math.ceil((rect.maxY-l.kH) / l.dH) + 1, rect.minY+1)
    end

  end
  return rect:snapToInt()
end -- inputToFeatureRect

function Localizer_3d:featureToInputRect(minX, minY, maxX, maxY, layer_index)
  layer_index = layer_index or #self.layers
  -- print('featureToInputRect: layer_index = ' .. layer_index) -- debug
  for i=layer_index,1,-1 do
    local l = self.layers[i]
    minX = minX * l.dW - l.padW
    minY = minY * l.dH - l.padW
    maxX = maxX * l.dW - l.padH + l.kW - l.dW
    maxY = maxY * l.dH - l.padH + l.kH - l.dH
  end
  return Rect.new(minX, minY, maxX, maxY)
end

