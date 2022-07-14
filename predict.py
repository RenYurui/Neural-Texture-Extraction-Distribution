""" Controllable person image synthesis using Cog"""
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import argparse
import os
import random
import tempfile

import torch
from cog import BaseModel, BasePredictor, Input, Path
from PIL import Image
from tqdm import tqdm

from config import Config
from data.demo_dataset import DemoDataset
from util.misc import to_cuda
from util.trainer import get_model_optimizer_and_scheduler, get_trainer, set_random_seed
from util.visualization import tensor2pilimage


def set_default_args():
    class Args:
        def __init__(self):
            pass

    args = Args()
    args.config = "./config/fashion_512.yaml"
    args.name = "fashion_512"
    args.checkpoints_dir = "result"  # dir for saving logs and models.')
    # self.args.seed = 0
    args.which_iter = 495400
    args.no_resume = True
    args.file_pairs = "./txt_files/demo.txt"
    args.output_dir = "./"
    args.input_dir = "./"
    return args


class Output(BaseModel):
    result: Path
    target_skeleton: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load model to make running multiple predictions more efficient"""

        print("Setting args......")
        self.args = set_default_args()
        self.opt = Config(self.args.config, self.args, is_train=False)
        self.opt.distributed = False
        self.opt.logdir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.device = torch.cuda.current_device()

        print("Loading models.......")
        # create a model
        (
            self.net_G,
            self.net_D,
            self.net_G_ema,
            self.opt_G,
            self.opt_D,
            self.sch_G,
            self.sch_D,
        ) = get_model_optimizer_and_scheduler(self.opt)

        trainer = get_trainer(
            self.opt,
            self.net_G,
            self.net_D,
            self.net_G_ema,
            self.opt_G,
            self.opt_D,
            self.sch_G,
            self.sch_D,
            None,
        )

        print("Loading model checkpoint......")
        current_epoch, current_iteration = trainer.load_checkpoint(
            self.opt, self.args.which_iter
        )
        self.net_G = trainer.net_G_ema.eval()

    def predict(
        self,
        reference_image: Path = Input(description="Input reference image"),
        desired_pose: Path = Input(
            description="Text file containing 2D array of Openpose keypoints (shape 18x2)"
        ),
        seed: int = Input(
            default=-1,
            description="Seed for random number generator. If -1, a random seed will be chosen. (minimum: -1; maximum: 4294967295)",
            ge=-1,
            le=(2**32 - 1),
        ),
    ) -> Output:

        # make it work with pngs
        reference_image = str(reference_image)
        im = Image.open(reference_image).convert("RGB")
        im.save(reference_image)

        seed = int(seed)
        if seed == -1:
            seed = random.randint(0, 2**32)
        set_random_seed(seed)
        print(f"Using seed {seed}......")

        os.makedirs(self.args.output_dir, exist_ok=True)
        data_root = (
            self.opt.data.path if self.args.input_dir is None else self.args.input_dir
        )
        data_loader = DemoDataset(data_root, self.opt.data, self.args.input_dir is None)

        print("Performing model inference.......")
        with torch.no_grad():
            data = data_loader.load_item(reference_image, str(desired_pose))
            data = to_cuda(data)
            # forward pass through model
            output = self.net_G(
                data["reference_image"],
                data["target_skeleton"],
            )
            fake_image = output["fake_image"][0]
            reference_image = data["reference_image"][0]
            target_skeleton = data["target_skeleton"][0, :3]

            # result = torch.cat([target_skeleton, fake_image], 2)

            # save resulting image
            result_path = Path(tempfile.mkdtemp()) / "result.png"
            result = tensor2pilimage(fake_image.clip(-1, 1), minus1to1_normalized=True)
            result.save(str(result_path))

            # save skeleton
            skeleton_path = Path(tempfile.mkdtemp()) / "skeleton.png"
            skeleton = tensor2pilimage(
                target_skeleton.clip(-1, 1), minus1to1_normalized=True
            )
            skeleton.save(str(skeleton_path))

            return Output(result=result_path, target_skeleton=skeleton_path)
