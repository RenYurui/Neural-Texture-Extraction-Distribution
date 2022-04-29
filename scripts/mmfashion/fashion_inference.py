import torch
import torch.nn as nn

from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose

import mmcv
from mmcv.parallel import collate, scatter
import numpy as np
import pycocotools.mask as maskUtils

class LoadImage(object):
    def __call__(self, results):
        if 'filename' not in results:
            results['filename'] = None
        results['img'] = results['img'][:,:,::-1] #RGB2BGR
        results['img_shape'] = results['img'].shape
        results['ori_shape'] = results['img'].shape
        return results  

class FashionInference(nn.Module):
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        super(FashionInference, self).__init__()
        self.seg_model = init_detector(
            config_path, checkpoint_path, device=device)

        self.cfg = self.seg_model.cfg
        
        self.test_pipeline = [LoadImage()] + self.cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(self.test_pipeline)
        self.device = device
        self.classes = self.seg_model.CLASSES
        # self.arg = arg

                
    def forward(self, image_array, find_items, filename=None, score_thr=0.3, ):
        # return_items: top, down, headwear, 
        data = dict(img=image_array, filename=filename)
        data = self.test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]
        # forward the model
        with torch.no_grad():
            result = self.seg_model(return_loss=False, rescale=True, **data)

        # Obtain Binary Mask
        bbox_result, segm_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]

        mask = 0
        for i in inds:
            i = int(i)
            label = self.classes[labels[i]]
            if label in self.mapping_classes(find_items):
                mask += maskUtils.decode(segms[i])

        return torch.tensor(mask > 0).float()[None, None]

    def mapping_classes(self, find_items):
        return_items=[]
        for item in find_items:
            if item == 'up':
                return_items.extend(['top','dress','outer'])

            elif item == 'down':
                return_items.extend(['leggings','skirt','pants','belt','footwear'])
            else:
                return_items.append(item)
        return return_items

# 'top', 'skirt', 'leggings', 'dress', 'outer', 'pants', 'bag',
# 'neckwear', 'headwear', 'eyeglass', 'belt', 'footwear', 'hair',
# 'skin', 'face'






      