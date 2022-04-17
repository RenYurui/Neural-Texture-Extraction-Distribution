import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnReconLoss(nn.Module):
    def __init__(self,  weights={8:1, 16:0.5, 32:0.25, 64:0.125, 128:0.0625},):
        super(AttnReconLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.weights = weights
    
    def forward(self, attn_dict, input_image, gt_image):
        softmax, query = attn_dict['extraction_softmax'], attn_dict['semantic_distribution']
        if isinstance(softmax, list) or isinstance(query, list):
            loss, weights = 0, 0
            for item_softmax, item_query in zip(softmax, query):
                h, w = item_query.shape[2:]
                gt_ = F.interpolate(gt_image, (h,w)).detach()
                input_ = F.interpolate(input_image, (h,w)).detach()
                estimated_target = self.cal_attn_image(
                    input_, item_softmax, item_query
                    )
                loss += self.l1loss(estimated_target, gt_) * self.weights[h]
                weights += self.weights[h]
            loss = loss/weights
        else:
            h, w = query.shape[2:]
            gt_ = F.interpolate(gt_image, (h,w))
            input_ = F.interpolate(input_image, (h,w))
            estimated_target = self.cal_attn_image(input_, softmax, query)
            loss = self.l1loss(estimated_target, gt_)
        return loss

    def cal_attn_image(self, input_image, softmax, query):
        b, num_label, h, w = query.shape
        if b != input_image.shape[0]:
            ib,ic,ih,iw = input_image.shape
            num_load_img = b // ib
            input_image = input_image[:,None].expand(ib, num_load_img, ic, ih, iw).contiguous()

        input_image = input_image.view(b, -1, h*w)
        extracted = torch.einsum('bkm,bvm->bvk', softmax, input_image)
        query = F.softmax(query.view(b, num_label, -1), 1)
        estimated_target = torch.einsum('bkm,bvk->bvm', query, extracted)
        estimated_target = estimated_target.view(b, -1, h, w)
        return estimated_target


        