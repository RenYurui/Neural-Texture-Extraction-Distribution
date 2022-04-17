import argparse

import data as Dataset
from config import Config
from util.logging import init_logging, make_logging_dir
from util.cudnn import init_cudnn
from util.distributed import init_dist
from util.distributed import master_only_print as print
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
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get training options
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=True)
    if args.debug:
        opt.data.train.batch_size=2


    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = opt.local_rank
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    init_cudnn(opt.cudnn.deterministic, opt.cudnn.benchmark)
    # create a dataset
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)

    # create a model
    net_G, net_D, net_G_ema, opt_G, opt_D, sch_G, sch_D \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_D, net_G_ema, \
                          opt_G, opt_D, sch_G, sch_D, \
                          train_dataset)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)   
    # training flag
    max_epoch = opt.max_epoch

    # Start training.
    for epoch in range(current_epoch, opt.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_dataset.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_dataset):
            data = trainer.start_of_iteration(data, current_iteration)
            trainer.optimize_parameters(data)
            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)
 
            if current_iteration >= opt.max_iter:
                print('Done with training!!!')
                break
        current_epoch += 1
        trainer.end_of_epoch(data, val_dataset, current_epoch, current_iteration)
