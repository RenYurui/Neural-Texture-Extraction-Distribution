import os
import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F 

from config import Config
from data.demo_appearance_dataset import DemoAppearanceDataset
from loss.perceptual import PerceptualLoss
from util.misc import to_cuda
from util.visualization import tensor2pilimage
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from third_part.mmdetection.fashion_inference import FashionInference


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/fashion_512.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int)
    parser.add_argument('--no_resume', action='store_true')

    parser.add_argument('--seg_config', 
                        default='./third_part/mmdetection/configs/mmfashion/mask_rcnn_r50_fpn_1x.py',)
    parser.add_argument('--seg_checkpoint', default='./third_part/mmdetection/epoch_15.pth',)    

    parser.add_argument('--output_dir', type=str, default='./demo_appearance_control')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--file_pairs', type=str, default='./txt_files/appearance_control.txt')
    parser.add_argument('--segment_parts', type=str, default='up')

    args = parser.parse_args()
    return args

def mask_select(query, mask_input):
    res_query = []
    for item in query:
        b,num_label,h,w = item.shape
        item = F.softmax(item.view(b, num_label, -1), 1)
        mask = F.interpolate(mask_input, (h,w)).bool().view(b,1,-1)
        item = torch.masked_select(item,  mask)
        item = torch.mean(item.view(b,num_label,-1), -1)
        res_query.append(item)
    return res_query


def max_pool_ref(query_list1, query_list2):
    query_list=[]
    for query1, query2 in zip(query_list1, query_list2):
        query, _ = torch.max(torch.cat([query1[:,:,None], query2[:,:,None]], 2), 2)
        query =  query[:,None,]
        query = (query >= (3.0 / query.shape[-1])).float()
        query_list.append(query)
    return query_list

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)
    opt.distributed = False
    opt.logdir = os.path.join(opt.checkpoints_dir, opt.name)
    opt.device = torch.cuda.current_device()
    opt.num_iteration = 200

    # create a model
    net_G, net_D, net_G_ema, opt_G, opt_D, sch_G, sch_D \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_D, net_G_ema, \
                          opt_G, opt_D, sch_G, sch_D, \
                          None)

    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)
    net_G = trainer.net_G_ema.eval()


    # define a segmentation model
    args.segment_parts = [item for item in args.segment_parts.split('-')]
    seg_model = FashionInference(args.seg_config, args.seg_checkpoint, device='cuda')

    # define dataset
    data_root = opt.data.path if args.input_dir is None else args.input_dir
    data_loader = DemoAppearanceDataset(data_root, opt.data, args.input_dir is None)
    garment_list, reference_list, skeleton_list= [],[],[]
    with open(args.file_pairs, 'r') as fd:
        files = fd.readlines()
        for file in files:
            garment,person,skeleton = file.replace('\n','').split(',')
            garment_list.append(garment)
            reference_list.append(person)
            skeleton_list.append(skeleton)

    # define loss 
    perceptual_loss = PerceptualLoss(
                network=opt.trainer.vgg_param.network,
                layers=opt.trainer.vgg_param.layers,
                num_scales=getattr(opt.trainer.vgg_param, 'num_scales', 1),
                ).to('cuda')
    os.makedirs(args.output_dir, exist_ok=True)

    # loop to generate the final results
    for garment_path, reference_path, skeleton_path in zip(garment_list, reference_list, skeleton_list):
        data = data_loader.load_item(garment_path, reference_path, skeleton_path)
        data = to_cuda(data)

        # init the interp coefficients
        with torch.no_grad():
            recoder_garment = collections.defaultdict(list)
            recoder_reference = collections.defaultdict(list)

            skeleton_feature = net_G.skeleton_encoder(data['target_skeleton'])

            _ = net_G.reference_encoder(data['garment_image'], recoder_garment)
            _ = net_G.reference_encoder(data['reference_image'], recoder_reference)
            neural_textures_garment = recoder_garment["neural_textures"]
            neural_textures_reference = recoder_reference["neural_textures"]

            garment_in_target_pose = net_G.target_image_renderer(
                skeleton_feature, neural_textures_garment, recoder_garment)
            person_in_target_pose = net_G.target_image_renderer(
                skeleton_feature, neural_textures_reference, recoder_reference)

            pil_garment = tensor2pilimage(garment_in_target_pose[0], minus1to1_normalized=True)
            mask_garment = seg_model(np.array(pil_garment), args.segment_parts).to('cuda')
            query_garment = mask_select(recoder_garment['semantic_distribution'], mask_garment)

            pil_reference = tensor2pilimage(person_in_target_pose[0], minus1to1_normalized=True)
            mask_reference = seg_model(np.array(pil_reference), args.segment_parts).to('cuda')
            query_reference = mask_select(recoder_reference['semantic_distribution'], mask_reference)

            interp_init = max_pool_ref(query_garment, query_reference)

        # optimize the interp coefficients
        interp=[]
        for item in interp_init:
            item.requires_grad = True
            interp.append(torch.nn.parameter.Parameter(item.to('cuda')))
        optimizer = optim.Adam(interp, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        for iterations in range(opt.num_iteration+1):
            neural_textures=[]
            
            for ext_garment, ext_reference, scale in zip(neural_textures_garment, neural_textures_reference, interp):
                neural_textures.append(ext_reference + (ext_garment-ext_reference)*scale)

            recoder = collections.defaultdict(list)
            output_images = net_G.target_image_renderer(
                skeleton_feature, neural_textures, recoder
                )
                
            if iterations >= 50 and iterations % 50 == 0:
                pil_out = tensor2pilimage(output_images.detach()[0], minus1to1_normalized=True)
                mask_out = seg_model(np.array(pil_out), args.segment_parts).to(output_images)
            else:
                mask_out = mask_garment

            querys_related = mask_select(recoder['semantic_distribution'], mask_out)
            querys_unrelated = mask_select(recoder['semantic_distribution'], 1-mask_out)
            
            regu_loss = 0
            for query_related, query_unrelated, scale in zip(querys_related, querys_unrelated, interp):
                query_related = (query_related >= (3.0 / query_related.shape[-1])).float()
                query_unrelated = (query_unrelated >= (3.0 / query_unrelated.shape[-1])).float()

                # Eq. 16                
                regu_loss += torch.mean(
                                   query_related.detach()*F.relu(1-scale) \
                                 + query_unrelated.detach()*F.relu(scale)
                                 )
            # Eq. 17 and Eq. 18  
            r1_loss = perceptual_loss(output_images*(1-mask_out), person_in_target_pose*(1-mask_reference))/torch.sum(1-mask_out)
            r2_loss = perceptual_loss(output_images*mask_out, garment_in_target_pose*mask_garment)/torch.sum(mask_out)
            
            total_loss = 300000*(10*r1_loss + r2_loss) + regu_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iterations % 50 == 0:
                image = torch.cat([
                    data['garment_image'],
                    data['reference_image'],
                    data['target_skeleton'][:,:3],
                    output_images], 3).clip(-1, 1)

                garment_path = os.path.splitext(os.path.basename(garment_path))[0]
                reference_path = os.path.splitext(os.path.basename(reference_path))[0]
                skeleton_path = os.path.splitext(os.path.basename(skeleton_path))[0]
                path = garment_path + '_2_' + reference_path + '_2_' + skeleton_path

                image = tensor2pilimage(image[0], minus1to1_normalized=True)
                image.save("./{}/{}_{}.png".format(args.output_dir,path,str(iterations)))
                print("save image to ./{}/{}_{}.png".format(args.output_dir,path,str(iterations)))
                print("Appearance Maintaining:{:4f}; Appearance Editing:{:4f}; Regularization:{:4f};".format(r1_loss,r2_loss,regu_loss))