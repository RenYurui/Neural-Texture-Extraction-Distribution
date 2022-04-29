wget -O ./mmfashion/__init__.py https://github.com/open-mmlab/mmfashion/raw/150f35454d94a0de7ae40dfdca7193207bd3fc57/configs/fashion_parsing_segmentation/__init__.py
wget -O ./mmfashion/mask_rcnn_r50_fpn_1x.py https://github.com/open-mmlab/mmfashion/raw/150f35454d94a0de7ae40dfdca7193207bd3fc57/configs/fashion_parsing_segmentation/mask_rcnn_r50_fpn_1x.py
wget -O ./mmfashion/mmfashion.py https://github.com/open-mmlab/mmfashion/raw/150f35454d94a0de7ae40dfdca7193207bd3fc57/configs/fashion_parsing_segmentation/mmfashion.py
gdown https://drive.google.com/uc?id=1q6zF7J6Gb-FFgM87oIORIt6uBozaXp5r
mv epoch_15.pth ../third_part/mmdetection/
mkdir ../third_part/mmdetection/configs/mmfashion/
cp ./mmfashion/mask_rcnn_r50_fpn_1x.py ../third_part/mmdetection/configs/mmfashion/
cp ./mmfashion/__init__.py ../third_part/mmdetection/mmdet/datasets
cp ./mmfashion/mmfashion.py ../third_part/mmdetection/mmdet/datasets
cp ./mmfashion/fashion_inference.py ../third_part/mmdetection/

# update some files in mmdetection
cp ./mmfashion/carafe_cuda.cpp ../third_part/mmdetection/mmdet/ops/carafe/src
cp ./mmfashion/carafe_naive_cuda.cpp ../third_part/mmdetection/mmdet/ops/carafe/src
cp ./mmfashion/roi_align_cuda.cpp ../third_part/mmdetection/mmdet/ops/roi_align/src
cp ./mmfashion/roi_pool_cuda.cpp ../third_part/mmdetection/mmdet/ops/roi_pool/src
cp ./mmfashion/deform_pool_cuda.cpp ../third_part/mmdetection/mmdet/ops/dcn/src
cp ./mmfashion/deform_conv_cuda.cpp ../third_part/mmdetection/mmdet/ops/dcn/src
cp ./mmfashion/masked_conv2d_cuda.cpp ../third_part/mmdetection/mmdet/ops/masked_conv/src
cp ./mmfashion/nms_cuda.cpp ../third_part/mmdetection/mmdet/ops/nms/src