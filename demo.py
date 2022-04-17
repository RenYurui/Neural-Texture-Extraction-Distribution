import os
import argparse
from tqdm import tqdm

import torch

from config import Config
from util.misc import to_cuda
from util.visualization import tensor2pilimage
from data.demo_dataset import DemoDataset
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/fashion_512.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--file_pairs', type=str, default='./demo.txt')
    parser.add_argument('--output_dir', type=str, default='./demo')
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get options
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)
    opt.distributed = False
    opt.logdir = os.path.join(opt.checkpoints_dir, opt.name)
    opt.device = torch.cuda.current_device()

    # create a model
    net_G, net_D, net_G_ema, opt_G, opt_D, sch_G, sch_D \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_D, net_G_ema, \
                          opt_G, opt_D, sch_G, sch_D, \
                          None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)                          
    net_G = trainer.net_G_ema.eval()

    reference_list, skeleton_list = [], []
    with open(args.file_pairs, 'r') as f:
        lines = f.readlines()
        for line in lines:
            reference,skeleton = line.replace('\n','').split(',')
            reference_list.append(reference)
            skeleton_list.append(skeleton)

    os.makedirs(args.output_dir, exist_ok=True)
    data_root = opt.data.path if args.input_dir is None else args.input_dir
    data_loader = DemoDataset(data_root, opt.data, args.input_dir is None)

    with torch.no_grad():
        for reference_path, skeleton_path in tqdm(zip(reference_list, skeleton_list)):
            data = data_loader.load_item(reference_path, skeleton_path)
            data = to_cuda(data)
            output = net_G(
                data['reference_image'], 
                data['target_skeleton'], 
            )
            fake_image = output['fake_image'][0]
            reference_image = data['reference_image'][0]
            target_skeleton = data['target_skeleton'][0,:3]
            reference_name = os.path.splitext(os.path.basename(reference_path))[0]
            skeleton_name = os.path.splitext(os.path.basename(skeleton_path))[0]
            name = '{}_2_{}.png'.format(reference_name, skeleton_name)
            result = torch.cat([reference_image,target_skeleton,fake_image], 2)
            tensor2pilimage(result.clip(-1, 1), minus1to1_normalized=True).save(
                os.path.join(args.output_dir, name)
            )