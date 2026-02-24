"""
Basic Command Line Interface, provides command line controls for training, test, and inference. Be sure to import this file before `import torch`, otherwise the OMP_NUM_THREADS would not work.
"""

import os

os.environ["OMP_NUM_THREADS"] = str(2)  # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ["MKL_NUM_THREADS"] = str(2)

from typing import *

import torch
import warnings
from model.utils import MyRichProgressBar as RichProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
from model.utils.my_logger import MyLogger as TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler

from pytorch_lightning.callbacks import (LearningRateMonitor, ModelSummary)
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.utilities.rank_zero import rank_zero_info

torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.


class BaseCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        self.add_model_invariant_arguments_to_parser(parser)

    def add_model_invariant_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # RichProgressBar
        if os.environ.get("CLEANMEL_DISABLE_RICH_PROGRESS", "0") != "1":
            parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
            parser.set_defaults({"progress_bar.console_kwargs": {
                "force_terminal": True,
                "no_color": True,
                "width": 200,
            }})

        # LearningRateMonitor
        parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "step",
        }
        parser.set_defaults(learning_rate_monitor_defaults)

        # ModelSummary
        parser.add_lightning_class_args(ModelSummary, 'model_summary')
        model_summary_defaults = {
            "model_summary.max_depth": 2,
        }
        parser.set_defaults(model_summary_defaults)

    def before_fit(self):
        profiler = AdvancedProfiler(dirpath="/nvmework3/shaonian/MelSpatialNet/MelSpatialNet/logs/OnlineSpatialNet/4xSPB_Hid96_offline/FLOPs_25.5G/Training_efficiency", filename="profile.txt")
        self.trainer.profiler = profiler
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # resume_from_checkpoint example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = self.model.name if hasattr(self.model, 'name') else type(self.model).__name__
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            ckpt_path = self.config['test']['model.ckpt_path']
            warnings.warn(f"You should give --ckpt_path if you want to test, currently using: {ckpt_path}")
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))

        torch.set_num_threads(5)

        test_set = 'test'
        if 'test_set' in self.config['test']['data']:
            test_set = self.config['test']['data']["test_set"]
        elif 'init_args' in self.config['test']['data'] and 'test_set' in self.config['test']['data']['init_args']:
            test_set = self.config['test']['data']['init_args']["test_set"]
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + test_set + '_set')

        self.copy_ckpt(exp_save_path=exp_save_path, ckpt_path=ckpt_path)

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)

    def before_predict(self):
        if self.config['predict']['ckpt_path']:
            ckpt_path = self.config['predict']['ckpt_path']
        else:
            ckpt_path = self.config['predict']['model.arch_ckpt']
            warnings.warn(f"You are not using lightning checkpoint in prediction, currently using: {ckpt_path}")
        try:
            exp_save_path = self.config['predict']["model.output_path"]
        except:
            exp_save_path = os.path.dirname(ckpt_path) + "/" + ckpt_path.split("/")[-1].split(".")[0] + "_inference_result"
        os.makedirs(exp_save_path, exist_ok=True)
        rank_zero_info(f"saving results to: {exp_save_path}")

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_predict(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for predict is removed: ' + self.trainer.log_dir + '/' + f)

    def copy_ckpt(self, exp_save_path: str, ckpt_path: str):
        # copy checkpoint to save path
        from pathlib import Path
        os.makedirs(exp_save_path, exist_ok=True)
        if (Path(exp_save_path) / Path(ckpt_path).parent).exists() == False:
            import shutil
            shutil.copyfile(ckpt_path, Path(exp_save_path) / Path(ckpt_path).parent)
