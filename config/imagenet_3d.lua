 local imgnet_cfg = {
  class_count = 30,  -- excluding background class
  target_smaller_side = 480, -- 600 in faster_rcnn Caffe
  scales = { 48, 96, 192, 384},
  max_pixel_size = 1000,
  normalization = { method = 'contrastive', width = 7, centering = true, scaling = true },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'rgb', -- 'yuv',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  background_base_path = '',
  batch_size = 300,
  positive_threshold = 0.6, 
  negative_threshold = 0.25,
  best_match = true,
  nearby_aversion = true,
  -- Added from faster_rcnn/functions/rpn/proposal_config
  feat_stride = 16,
  image_means = '/home/bingbin/faster-rcnn.torch/image_means.txt',
  test_scales = 600,
  test_max_size = 1000,
  test_min_box_size = 16,
  -- Added on Mar 27th: load Caffe model
  -- pretrain_prototxt = 'models/prototxt/solver_60k80k.prototxt',
  -- pretrain_model = 'models/pretrain_models/ZF.caffemodel'
}

return imgnet_cfg
