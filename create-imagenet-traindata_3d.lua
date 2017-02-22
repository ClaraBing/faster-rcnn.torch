-- specify the base path of the ILSVRC2015 dataset: 
ILSVRC2015_BASE_DIR = '/disk/bingbin/ILSVRC2015_sampled_double/' -- 65817 imgs for training

require 'lfs'         -- Lua file system
require 'LuaXML'      -- if missing use luarocks install LuaXML
require 'utilities'
require 'Rect' 

local ground_truth = {}
local class_names = {}
local class_index = {}

local file_cnt = 0

function import_file(anno_base, data_base, fn, name_table)
  print('import_file: ' .. fn)
  -- Modified on Feb 19th: add fn names beforehand s.t. can use shuffle
  local fidx = tonumber(fn:split('/')[9]:sub(1,6))
  local prev_fidx, next_fidx = fidx-3, fidx+3
  local fn_prefix = fn:sub(1, -11)
  local prev_fn = fn_prefix .. string.format('%06d', fidx-3) .. '.xml'
  local prev_prev_fn = fn_prefix .. string.format('%06d', fidx-6) .. '.xml'
  local next_fn = fn_prefix .. string.format('%06d', fidx+3) .. '.xml'
  local next_next_fn = fn_prefix .. string.format('%06d', fidx+6) .. '.xml'
  if (io.open(prev_fn, 'r') == nil and io.open(prev_prev_fn, 'r') ~= nil) or (io.open(next_fn, 'r') == nil and io.open(next_next_fn, 'r') ~= nil) then
    return
  end

  if io.open(prev_fn, 'r') == nil then        
    prev_fn = fn
    print('Amend prev_fn for ' .. fn)
  end
  if io.open(next_fn, 'r') == nil then
    next_fn = fn
    print('Amend next_fn for ' .. fn)
  end

  local file_seq = {[1]=prev_fn, [2]=fn, [3]=next_fn}
  local img_seq = {}
  -- local rois_seq = {}
  local curr_fn
  for i=1,3 do
    curr_fn = file_seq[i]
    local x = xml.load(fn)
    local a = x:find('annotation')
    local folder = a:find('folder')[1]
    local filename = a:find('filename')[1]
    local src = a:find('source')
    local db = src:find('database')[1]
    local sz = a:find('size')
    local w = tonumber(sz:find('width')[1])
    local h = tonumber(sz:find('height')[1])
  
    for _,e in pairs(a) do
      if e[e.TAG] == 'object' then 
        local obj = e
        local name = obj:find('name')[1]
        local bb = obj:find('bndbox') 
        local xmin = tonumber(bb:find('xmin')[1])
        local xmax = tonumber(bb:find('xmax')[1])
        local ymin = tonumber(bb:find('ymin')[1])
        local ymax = tonumber(bb:find('ymax')[1])
        
        if not class_index[name] then
          class_names[#class_names + 1] = name
          class_index[name] = #class_names 
        end 
        
  --        local fn_lst = {[1]=prev_fn, [2]=fn, [3]=next_fn}
        img_seq[i] = path.join(data_base, path.relpath(curr_fn, anno_base))
        -- replace 'xml' file ending with 'JPEG'
        img_seq[i] = string.sub(img_seq[i], 1, #img_seq[i] - 3) .. 'JPEG'    
    
          
        local roi = {
          rect = Rect.new(xmin, ymin, xmax, ymax),
          class_index = class_index[name],
          class_name = name
        }
        -- rois_seq[i] = roi
        
  -- Mark
        local file_entry = ground_truth[img_seq[i]]
        if not file_entry then
          file_entry = { image_file_name = img_seq[i], rois = {} }
          ground_truth[img_seq[i]] = file_entry -- use the middle img as index
        end
        table.insert(file_entry.rois, roi)
      end -- if e[e.TAG] == 'object'
    end -- for _,e in pairs(a) do
  end -- for i=1,3
  table.insert(name_table, img_seq[2])
  print('  successfully imported:' .. img_seq[2])
end -- function 'import_file'

function import_directory(anno_base, data_base, directory_path, recursive, name_table)
   print('import dir: ' .. directory_path)
   for fn in lfs.dir(directory_path) do
    local full_fn = path.join(directory_path, fn)
    local mode = lfs.attributes(full_fn, 'mode') 
    if recursive and mode == 'directory' and fn ~= '.' and fn ~= '..' then
      import_directory(anno_base, data_base, full_fn, true, name_table)
      collectgarbage()
    elseif mode == 'file' and string.sub(fn, -4):lower() == '.xml' then
      import_file(anno_base, data_base, full_fn, name_table)
      file_cnt = file_cnt + 1
      -- return -- Mark
    end
    if #ground_truth > 10 then
      print('ground truth files > 10.\nReturning.')
      return
    end
  end
  print('\n\n')
  return l
end

-- recursively search through training and validation directories and import all xml files
function create_ground_truth_file(dataset_name, base_dir, train_annotation_dir, val_annotation_dir, train_data_dir, val_data_dir, background_dirs, output_fn)
  function expand(p)
    return path.join(base_dir, p)
  end
  
  local training_set, validation_set = {}, {}
  import_directory(expand(train_annotation_dir), expand(train_data_dir), expand(train_annotation_dir), true, training_set)
  print('\n\n\nTraining data ready\n\n\n')
  import_directory(expand(val_annotation_dir), expand(val_data_dir), expand(val_annotation_dir), true, validation_set)
  print('\n\n\nValidation data ready\n\n\n')
  local file_names = keys(ground_truth)

  -- compile list of background images -- Note by Bingbin: ignored here
  local background_files = {}
 
  print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d; (Background: %d)', 
    #file_names, #class_names, #training_set, #validation_set, #background_files
  ))
  save_obj(
    output_fn,
    {
      dataset_name = dataset_name,
      ground_truth = ground_truth,
      training_set = training_set,
      validation_set = validation_set,
      class_names = class_names,
      class_index = class_index,
      background_files = background_files
    }
  )
  print('Done. \nFile count = ' .. file_cnt)
end -- create_ground_truth_file


background_folders = {}
for i=0,10 do
  table.insert(background_folders, 'Data/VID/train/' .. i)
end

create_ground_truth_file(
  'ILSVRC2015_VID',
  ILSVRC2015_BASE_DIR,
  'Annotations/VID/train', 
  'Annotations/VID/val', -- Note: validation data is also from VID train set
  'Data/VID/train',
  'Data/VID/val',
  background_folders,
  -- 'ILSVRC2015_VID_test.t7'
  'data_mine/ILSVRC2015_VID_sampled_double_mid_only.t7'
)
