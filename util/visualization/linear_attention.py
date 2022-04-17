import torch
import torch.nn.functional as F


def attn2image(source_softmax, seg_query, input_image):
    target_list = []
    image_size = input_image.shape[2:]
    for softmax, query in zip(source_softmax, seg_query):
        b, num_label, h, w = query.shape
        input_resize = F.interpolate( input_image, (h,w) )
        input_resize = input_resize.view(b, -1, h*w)
        extracted = torch.einsum('bkm,bvm->bvk', softmax, input_resize)
        query = F.softmax(query.view(b, num_label, -1), 1)
        estimated_target = torch.einsum('bkm,bvk->bvm', query, extracted)
        estimated_target = estimated_target.view(b, -1, h, w)
        target_list.append(F.interpolate(estimated_target, image_size))
    target_gen = torch.cat(target_list, 3)
    return target_gen 