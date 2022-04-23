from email.policy import strict
import os
import glob
import time

import torch
import torchvision
from torch import nn

from util.distributed import is_master, master_only
from util.distributed import master_only_print as print
from util.meters import Meter, add_hparams
from util.misc import to_cuda

class BaseTrainer(object):
    def __init__(self,
                 opt,
                 net_G,
                 net_D,
                 net_G_ema,
                 opt_G,
                 opt_D,
                 sch_G,
                 sch_D,
                 train_data_loader,
                 val_data_loader=None):
        super(BaseTrainer, self).__init__()
        print('Setup trainer.')

        # Initialize models and data loaders.
        self.opt = opt
        self.net_G = net_G
        if opt.distributed:
            self.net_G_module = self.net_G.module
        else:
            self.net_G_module = self.net_G

        self.is_inference = train_data_loader is None
        self.net_D = net_D
        self.net_G_ema = net_G_ema
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.sch_G = sch_G
        self.sch_D = sch_D
        self.train_data_loader = train_data_loader

        self.criteria = nn.ModuleDict()
        self.weights = dict()
        self.losses = dict(gen_update=dict(), dis_update=dict())
        self.gen_losses = self.losses['gen_update']
        self.dis_losses = self.losses['dis_update']
        self._init_loss(opt)
        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and \
                    self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

        if self.is_inference:
            return

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self.elapsed_iteration_time = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.time_iteration = -1
        self.time_epoch = -1
                
        # Initialize tensorboard and hparams.
        self._init_tensorboard()
        self._init_hparams()

    def _init_tensorboard(self):
        # Logging frequency: self.opt.logging_iter
        self.meters = {}
        names = ['optim/gen_lr', 'optim/dis_lr', 'time/iteration', 'time/epoch']
        for name in names:
            self.meters[name] = Meter(name)

        # Logging frequency: self.opt.image_display_iter
        self.image_meter = Meter('images')

    def _init_hparams(self):
        self.hparam_dict = {}

    def _write_tensorboard(self):
        self._write_to_meters({'time/iteration': self.time_iteration,
                               'time/epoch': self.time_epoch,
                               'optim/gen_lr': self.sch_G.get_last_lr()[0],
                               'optim/dis_lr': self.sch_D.get_last_lr()[0]},
                              self.meters)
        self._write_loss_meters()
        self._write_custom_meters()
        self._flush_meters(self.meters)

    def _write_loss_meters(self):
        for loss_name, loss in self.gen_losses.items():
            full_loss_name = 'gen_update' + '/' + loss_name
            if full_loss_name not in self.meters.keys():
                self.meters[full_loss_name] = Meter(full_loss_name)
            self.meters[full_loss_name].write(loss.item())

        for loss_name, loss in self.dis_losses.items():
            full_loss_name = 'dis_update' + '/' + loss_name
            if full_loss_name not in self.meters.keys():
                self.meters[full_loss_name] = Meter(full_loss_name)
            self.meters[full_loss_name].write(loss.item())

    def _write_custom_meters(self):
        pass

    @staticmethod
    def _write_to_meters(data, meters):
        for key, value in data.items():
            meters[key].write(value)

    def _flush_meters(self, meters):
        for meter in meters.values():
            meter.flush(self.current_iteration)

    def _pre_save_checkpoint(self):
        pass

    def save_checkpoint(self, current_epoch, current_iteration):
        self._pre_save_checkpoint()
        _save_checkpoint(self.opt,
                         self.net_G, self.net_D, self.net_G_ema,
                         self.opt_G, self.opt_D,
                         self.sch_G, self.sch_D,
                         current_epoch, current_iteration)

    def load_checkpoint(self, opt, which_iter=None):
        if which_iter is not None:
            model_path = os.path.join(
                opt.logdir, '*_iteration_{:09}_checkpoint.pt'.format(which_iter))
            latest_checkpoint_path = glob.glob(model_path)
            assert len(latest_checkpoint_path) <= 1, "please check the saved model {}".format(
                model_path)
            if len(latest_checkpoint_path) == 0:
                current_epoch = 0
                current_iteration = 0
                print('No checkpoint found at iteration {}.'.format(which_iter))
                return current_epoch, current_iteration
            checkpoint_path = latest_checkpoint_path[0]

        elif os.path.exists(os.path.join(opt.logdir, 'latest_checkpoint.txt')):
            with open(os.path.join(opt.logdir, 'latest_checkpoint.txt'), 'r') as f:
                line = f.readlines()[0].replace('\n', '')
                checkpoint_path = os.path.join(opt.logdir, line.split(' ')[-1])
        else:
            current_epoch = 0
            current_iteration = 0
            print('No checkpoint found.')
            return current_epoch, current_iteration
        resume = opt.phase == 'train' and opt.resume
        current_epoch, current_iteration = self._load_checkpoint(
            checkpoint_path, resume)
        return current_epoch, current_iteration


    def _load_checkpoint(self, checkpoint_path, resume=True):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.net_G_ema.load_state_dict(checkpoint['net_G_ema'])
        print('load [net_G_ema] from {}'.format(checkpoint_path))
        if self.opt.phase == 'train':
            if 'net_G' not in checkpoint:
                self.net_G.module.load_state_dict(checkpoint['net_G_ema']) 
                print('load_from_net_ema')
            else:
                self.net_G.load_state_dict(checkpoint['net_G'])  
            self.net_D.load_state_dict(checkpoint['net_D'])
            print('load [net_G] and [net_D] from {}'.format(checkpoint_path))
            if resume:
                self.opt_G.load_state_dict(checkpoint['opt_G'])
                self.opt_D.load_state_dict(checkpoint['opt_D'])
                self.sch_G.load_state_dict(checkpoint['sch_G'])
                self.sch_D.load_state_dict(checkpoint['sch_D'])
                print('load optimizers and schdules from {}'.format(checkpoint_path))


        if resume or self.opt.phase == 'test':
            current_epoch = checkpoint['current_epoch']
            current_iteration = checkpoint['current_iteration']
        else:
            current_epoch = 0
            current_iteration = 0
        print('Done with loading the checkpoint.')
        return current_epoch, current_iteration  

        
    def start_of_epoch(self, current_epoch):
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def end_of_epoch(self, data, val_dataset, current_epoch, current_iteration):
        # Update the learning rate policy for the generator if operating in the
        # epoch mode.
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.opt.gen_optimizer.lr_policy.iteration_mode:
            self.sch_G.step()
        # Update the learning rate policy for the discriminator if operating
        # in the epoch mode.
        if not self.opt.dis_optimizer.lr_policy.iteration_mode:
            self.sch_D.step()
        elapsed_epoch_time = time.time() - self.start_epoch_time
        # Logging.
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch,
                                                     elapsed_epoch_time))
        self.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_epoch >= self.opt.snapshot_save_start_epoch and \
                current_epoch % self.opt.snapshot_save_epoch == 0:
            self.save_image(self._get_save_path('image', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)

    def start_of_iteration(self, data, current_iteration):
        data = self._start_of_iteration(data, current_iteration)
        data = to_cuda(data)
        self.current_iteration = current_iteration
        if not self.is_inference:
            self.net_D.train()
            self.net_G.train()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        # Update the learning rate policy for the generator if operating in the
        # iteration mode.
        if self.opt.gen_optimizer.lr_policy.iteration_mode:
            self.sch_G.step()
        # Update the learning rate policy for the discriminator if operating
        # in the iteration mode.
        if self.opt.dis_optimizer.lr_policy.iteration_mode:
            self.sch_D.step()

        # Accumulate time
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % self.opt.logging_iter == 0:
            ave_t = self.elapsed_iteration_time / self.opt.logging_iter
            self.time_iteration = ave_t
            print('Iteration: {}, average iter time: '
                  '{:6f}.'.format(current_iteration, ave_t))
            self.elapsed_iteration_time = 0

        self._end_of_iteration(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_iteration >= self.opt.snapshot_save_start_iter and \
                current_iteration % self.opt.snapshot_save_iter == 0:
            self.save_image(self._get_save_path('image', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)
        # Compute image to be saved.
        elif current_iteration % self.opt.image_save_iter == 0:
            self.save_image(self._get_save_path('image', 'jpg'), data)

        if current_iteration % self.opt.logging_iter == 0:
            self._write_tensorboard()
            self._print_current_errors()


    def _print_current_errors(self):
        epoch, iteration = self.current_epoch, self.current_iteration
        message = '(epoch: %d, iters: %d) ' % (epoch, iteration)
        for loss_name, losses in self.gen_losses.items():
            full_loss_name = 'gen_update' + '/' + loss_name
            message += '%s: %.3f ' % (full_loss_name, losses)

        for loss_name, losses in self.dis_losses.items():
            full_loss_name = 'dis_update' + '/' + loss_name
            message += '%s: %.3f ' % (full_loss_name, losses)

        print(message)
        log_name = os.path.join(self.opt.logdir, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def write_data_tensorboard(self, data, epoch, iteration):
        for name, value in data.items():
            full_name = 'eval/' + name
            if full_name not in self.meters.keys():
                # Create a new meter if it doesn't exist.
                self.meters[full_name] = Meter(full_name)
            self.meters[full_name].write(value)
            self.meters[full_name].flush(iteration)

    def save_image(self, path, data):
        self.net_G.eval()
        vis_images = self._get_visualizations(data)
        if is_master() and vis_images is not None:
            # vis_images = torch.cat(vis_images, dim=3).float()
            vis_images = (vis_images + 1) / 2
            print('Save output images to {}'.format(path))
            vis_images.clamp_(0, 1)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image_grid = torchvision.utils.make_grid(
                vis_images, nrow=1, padding=0, normalize=False)
            if self.opt.trainer.image_to_tensorboard:
                self.image_meter.write_image(image_grid, self.current_iteration)
            torchvision.utils.save_image(image_grid, path, nrow=1)


    def _get_save_path(self, subdir, ext):
        subdir_path = os.path.join(self.opt.logdir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(
            subdir_path, 'epoch_{:05}_iteration_{:09}.{}'.format(
                self.current_epoch, self.current_iteration, ext))

    def _start_of_epoch(self, current_epoch):
        pass

    def _start_of_iteration(self, data, current_iteration):
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        pass
    
    def _end_of_epoch(self, data, current_epoch, current_iteration):
        pass

    def _get_visualizations(self, data):
        return None

    def _init_loss(self, opt):
        raise NotImplementedError

    def optimize_parameters(self, data):
        raise NotImplementedError

    def test(self, data_loader, output_dir, current_iteration):
        raise NotImplementedError

@master_only
def _save_checkpoint(opt,
                     net_G, net_D, net_G_ema,
                     opt_G, opt_D,
                     sch_G, sch_D,
                     current_epoch, current_iteration):
    latest_checkpoint_path = 'epoch_{:05}_iteration_{:09}_checkpoint.pt'.format(
        current_epoch, current_iteration)
    save_path = os.path.join(opt.logdir, latest_checkpoint_path)
    torch.save(
        {
            'net_G': net_G.state_dict(),
            'net_D': net_D.state_dict(),
            'net_G_ema': net_G_ema.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'sch_G': sch_G.state_dict(),
            'sch_D': sch_D.state_dict(),
            'current_epoch': current_epoch,
            'current_iteration': current_iteration,
        },
        save_path,
    )
    fn = os.path.join(opt.logdir, 'latest_checkpoint.txt')
    with open(fn, 'wt') as f:
        f.write('latest_checkpoint: %s' % latest_checkpoint_path)
    print('Save checkpoint to {}'.format(save_path))
    return save_path