-- specify the base path of the ILSVRC2015 dataset: 
-- ILSVRC2015_BASE_DIR = '/disk/bingbin/ILSVRC2015_sampled_double/'
ILSVRC2015_BASE_DIR = '/disk/bingbin/VIDdevkit/ILSVRC2015_test/'
imdb_save_name = 'data_mine/ILSVRC2015_test_fast_rcnn.t7'

require 'lfs'         -- Lua file system
require 'LuaXML'      -- if missing use luarocks install LuaXML
require 'utilities'
require 'Rect' 

local ground_truth = {}
local class_names = {}
local class_index = {}
-- Added by Bingbin
local image_ids = {}

function import_file(anno_base, data_base, fn, name_table)
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
      
      -- generate path relative to annotation dir and join with data dir
      local image_path = path.join(data_base, path.relpath(fn, anno_base))  
      
      -- replace 'xml' file ending with 'JPEG'
      image_path = string.sub(image_path, 1, #image_path - 3) .. 'JPEG'    
      
      table.insert(name_table, image_path)
      
      local roi = {
        rect = Rect.new(xmin, ymin, xmax, ymax),
        class_index = class_index[name],
        class_name = name
      }
      
      local file_entry = ground_truth[image_path]
      if not file_entry then
        file_entry = { image_file_name = image_path, rois = {} }
        table.insert(image_ids, image_path)
        ground_truth[image_path] = file_entry
      end
      -- print(file_entry) -- Mark
      table.insert(file_entry.rois, roi)
    end
  end
end

-- NOTE: this function will import all files in a directory, rather than following the ImageSet file
function import_directory(imgset_file, anno_base, data_base, directory_path, recursive, name_table)
   local file_ids = lines_from(imgset_file)
   print('imgset_file: ' .. imgset_file)
   print('#file_ids: ' .. #file_ids)
   -- for fn in lfs.dir(directory_path) do
   for i,fn in ipairs(file_ids) do
    local full_fn = path.join(directory_path, fn..'.xml')
    local mode = lfs.attributes(full_fn, 'mode') 
    if mode == nil then
        mode = 'file'
    end
    if recursive and mode == 'directory' and fn ~= '.' and fn ~= '..' then
      print('recursive call to import_diretory')
      import_directory(anno_base, data_base, full_fn, true, name_table)
      collectgarbage()
    elseif mode == 'file' and string.sub(full_fn, -4):lower() == '.xml' then
      import_file(anno_base, data_base, full_fn, name_table)
    end
    if #ground_truth > 10 then
      print('ground truth files > 10.\nReturning.')
      return
    end
  end
  return l
end

-- recursively search through training and validation directories and import all xml files
function create_ground_truth_file(dataset_name, base_dir, train_imgset_file, val_imgset_file, train_annotation_dir, val_annotation_dir, train_data_dir, val_data_dir, background_dirs, output_fn)
  function expand(p)
    return path.join(base_dir, p)
  end
  
  local training_set = {}
  local validation_set = {}
  import_directory(path.join(base_dir, train_imgset_file), expand(train_annotation_dir), expand(train_data_dir), expand(train_annotation_dir), true, training_set)
  import_directory(path.join(base_dir, val_imgset_file), expand(val_annotation_dir), expand(val_data_dir), expand(val_annotation_dir), true, validation_set)
  local file_names = keys(ground_truth)
  
  -- compile list of background images
  local background_files = {}
--  for i,directory_path in ipairs(background_dirs) do
--    directory_path = expand(directory_path)
--    for fn in lfs.dir(directory_path) do
--      local full_fn = path.join(directory_path, fn)
--      local mode = lfs.attributes(full_fn, 'mode')
--      if mode == 'file' and string.sub(fn, -5):lower() == '.jpeg' then
--        table.insert(background_files, full_fn)
--      end
--    end
--  end
  
  print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d; (Background: %d)', 
    #file_names, #class_names, #training_set, #validation_set, #background_files
  ))
  -- image_ids = torch.Tensor(image_ids) -- Modified on Mar 25th: keep it as table since entries are not numbers
  save_obj(
    output_fn,
    {
      dataset_name = dataset_name,
      ground_truth = ground_truth,
      training_set = training_set,
      validation_set = validation_set,
      class_names = class_names,
      class_index = class_index,
      image_ids = image_ids,
      background_files = background_files
    }
  )
  print('Done.')
end


background_folders = {}
for i=0,10 do
  table.insert(background_folders, 'Data/VID/train/' .. i)
end

create_ground_truth_file(
  'ILSVRC2015_VID',
  ILSVRC2015_BASE_DIR,
  'ImageSets/VID/uniq_train.txt', -- Added by Bingbin: import only files specified in ImageSet files
  'ImageSets/VID/val.txt',
  'Annotations/VID', -- 'Annotations/VID/train', 
  'Annotations/VID', -- 'Annotations/VID/val', -- Note: validation data is also from VID train set
  'Data/VID', -- 'Data/VID/train',
  'Data/VID', -- 'Data/VID/val',
  background_folders,
  imdb_save_name
  -- 'ILSVRC2015_VID_test.t7'
  -- 'data_mine/ILSVRC2015_VID_sampled_double.t7'
)
