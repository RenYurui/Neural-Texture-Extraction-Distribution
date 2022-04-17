import collections
from torch import nn
from generators.base_module import Encoder, Decoder

class Generator(nn.Module):
    def __init__(
        self,
        size,
        semantic_dim,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.size = size
        self.reference_encoder = Encoder(
            size, 3, channels, num_labels, match_kernels, blur_kernel
        )
            
        self.skeleton_encoder = Encoder(
            size, semantic_dim, channels, 
            )

        self.target_image_renderer = Decoder(
            size, channels, num_labels, match_kernels, blur_kernel
        )

    def _cal_temp(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(
        self,
        source_image,
        skeleton,
    ):
        output_dict={}
        recoder = collections.defaultdict(list)
        skeleton_feature = self.skeleton_encoder(skeleton)
        _ = self.reference_encoder(source_image, recoder)
        neural_textures = recoder["neural_textures"]
        output_dict['fake_image'] = self.target_image_renderer(
            skeleton_feature, neural_textures, recoder
            )
        output_dict['info'] = recoder
        return output_dict

