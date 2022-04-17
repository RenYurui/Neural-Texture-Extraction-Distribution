import os
import math
import importlib
from tqdm import tqdm

import torch
from torch import autograd

from loss.perceptual  import PerceptualLoss
from loss.gan import GANLoss
from loss.attn_recon import AttnReconLoss
from util.visualization import attn2image, tensor2pilimage
from util.trainer import accumulate
from trainers.base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, opt, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader=None):
        super(Trainer, self).__init__(opt, net_G, net_D, opt_G,
                                      opt_D, sch_G, sch_D,
                                      train_data_loader, val_data_loader)
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        self.seg_to_color = {}

        if getattr(self.opt.trainer, 'face_crop_method', None):
            file, crop_func = self.opt.trainer.face_crop_method.split('::')
            file = importlib.import_module(file)
            self.crop_func = getattr(file, crop_func)

    def _init_loss(self, opt):
        r"""Define training losses.

        Args:
            opt: options defined in yaml file.
        """        
        self._assign_criteria(
            'perceptual',
            PerceptualLoss(
                network=opt.trainer.vgg_param.network,
                layers=opt.trainer.vgg_param.layers,
                num_scales=getattr(opt.trainer.vgg_param, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual)

        self._assign_criteria(
            'attn_rec',
            AttnReconLoss(opt.trainer.attn_weights).to('cuda'),
            opt.trainer.loss_weight.weight_attn_rec)

        self._assign_criteria(
            'gan',
            GANLoss(opt.trainer.gan_mode).to('cuda'),
            opt.trainer.loss_weight.weight_gan)   
        
        if getattr(opt.trainer.loss_weight, 'weight_face', 0) != 0:
            self._assign_criteria(
                'face', 
                PerceptualLoss(
                    network=opt.trainer.vgg_param.network,
                    layers=opt.trainer.vgg_param.layers,
                    num_scales=1,
                    ).to('cuda'),
                opt.trainer.loss_weight.weight_face)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data):
        r"""Training step of generator and discriminator

        Args:
            data (dict): data used in the training step
        """          
        # training step of the generator        
        self.gen_losses = {}
        source_image, target_image = data['source_image'], data['target_image']
        source_skeleton, target_skeleton = data['source_skeleton'], data['target_skeleton']

        input_image = torch.cat((source_image, target_image), 0)
        input_skeleton = torch.cat((target_skeleton, source_skeleton), 0)
        gt_image = torch.cat((target_image, source_image), 0) 

        output_dict = self.net_G(input_image, input_skeleton)
        fake_img, info = output_dict['fake_image'], output_dict['info']

        if self.cal_gan_flag:
            fake_pred = self.net_D(fake_img)
            g_loss = self.criteria['gan'](fake_pred, t_real=True, dis_update=False)
            self.gen_losses["gan"] = g_loss 
        else:
            self.gen_losses["gan"] = torch.tensor(0.0, device='cuda')

        self.gen_losses["perceptual"] = self.criteria['perceptual'](fake_img, gt_image)
        self.gen_losses['attn_rec'] = self.criteria['attn_rec'](info, input_image, gt_image)

        if 'target_face_center' in data and 'face' in self.criteria:
            source_face_center, target_face_center  = data['source_face_center'], data['target_face_center']  
            target_face_center = torch.cat((target_face_center, source_face_center), 0)
            self.gen_losses['face'] = self.criteria['face'](
                self.crop_func(fake_img,
                               target_face_center),
                self.crop_func(gt_image,
                               target_face_center))    
        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)

        # training step of the discriminator
        if self.cal_gan_flag:
            self.dis_losses = {}
            fake_pred = self.net_D(fake_img.detach())
            real_pred = self.net_D(gt_image)
            fake_loss = self.criteria['gan'](fake_pred, t_real=False, dis_update=True)
            real_loss = self.criteria['gan'](real_pred, t_real=True,  dis_update=True)
            d_loss = fake_loss + real_loss
            self.dis_losses["d"] = d_loss
            self.dis_losses["real_score"] = real_pred.mean()
            self.dis_losses["fake_score"] = fake_pred.mean()        

            self.net_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if self.d_regularize_flag:
                gt_image.requires_grad = True
                real_img_aug = gt_image
                real_pred = self.net_D(real_img_aug)
                r1_loss = self.d_r1_loss(real_pred, gt_image)

                self.net_D.zero_grad()
                (self.opt.trainer.r1 / 2 * r1_loss * self.opt.trainer.d_reg_every + 0 * real_pred[0]).backward()

                self.opt_D.step()

                self.dis_losses["r1"] = r1_loss

    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def _start_of_iteration(self, data, current_iteration):
        r"""processing before iteration

        Args:
            data (dict): data used in the training step
            current_iteration (int): current iteration 
        """             
        self.cal_gan_flag = current_iteration > self.opt.trainer.gan_start_iteration
        self.d_regularize_flag = current_iteration % self.opt.trainer.d_reg_every == 0
        return data

    def _get_visualizations(self, data):
        r"""save visualizations when training the model

        Args:
            data (dict): data used in the training step
        """          
        source_image, target_image = data['source_image'], data['target_image']
        source_skeleton, target_skeleton = data['source_skeleton'], data['target_skeleton']
        input_image = torch.cat((source_image, target_image), 0)
        input_skeleton = torch.cat((target_skeleton, source_skeleton), 0)

        with torch.no_grad():
            self.net_G_ema.eval()
            output_dict = self.net_G_ema(
                input_image, input_skeleton)

            fake_img_rec, info = output_dict['fake_image'], output_dict['info']
            fake_target_rec, fake_source_rec = torch.chunk(fake_img_rec, 2, dim=0)

            attn_image = attn2image(info['hook_softmax'], info['semantic_distribution'], input_image)
            attn_target, attn_source = torch.chunk(attn_image, 2, dim=0)
            sample1 = torch.cat([source_image.cpu(), source_skeleton[:,:3].cpu(), fake_source_rec.cpu(), attn_source.cpu()], 3)
            sample2 = torch.cat([target_image.cpu(), target_skeleton[:,:3].cpu(), fake_target_rec.cpu(), attn_target.cpu()], 3)
            sample = torch.cat([sample1, sample2], 2)
            sample = torch.cat(torch.chunk(sample, source_image.size(0), 0)[:3], 2)
        return sample



    def test(self, data_loader, output_dir, current_iteration=-1):
        r"""inference function

        Args:
            data_loader: dataloader of the dataset
            output_dir (str): folder for saving the result images
            current_iteration (int): current iteration 
        """                  
        net_G = self.net_G_ema.eval()
        os.makedirs(output_dir, exist_ok=True)
        print('number of samples %d' % len(data_loader))
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration)
            input_skeleton = data['target_skeleton']
            input_image = data['source_image']
            with torch.no_grad():
                output_dict = net_G(
                    input_image, input_skeleton)    
            output_images = output_dict['fake_image']
            for output_image, file_name in zip(output_images, data['path']):
                fullname = os.path.join(output_dir, file_name)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                output_image.save(fullname)
        return None

