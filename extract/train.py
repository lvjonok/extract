import matplotlib

matplotlib.use("Agg")
import torch
import os
import time
from shutil import copy
import datetime
import imp

# from tensorboardX import SummaryWriter
import numpy as np
import random
from torch import autograd
from torch.optim import Adam, RMSprop, SGD, AdamW
from functools import partial

from extract.components.data_loader import RandomVideoDataset
from extract.utils.general_utils import RecursiveAverageMeter, map_dict, timing
from extract.rl.components.replay_buffer import RolloutStorage
from extract.rl.components.sampler import Sampler
from extract.components.checkpointer import (
    CheckpointHandler,
    save_cmd,
    save_git,
    get_config_path,
)
from extract.utils.general_utils import (
    dummy_context,
    AttrDict,
    get_clipped_optimizer,
    AverageMeter,
    ParamDict,
)
from extract.utils.pytorch_utils import (
    LossSpikeHook,
    NanGradHook,
    NoneGradHook,
    DataParallelWrapper,
    RAdam,
)
from extract.components.trainer_base import BaseTrainer
from extract.utils.wandb import WandBLogger
from extract.components.params import get_args

WANDB_PROJECT_NAME = "p-amazon-intern"
WANDB_ENTITY_NAME = "jesbu1"
WANDB_PROJECT_NAME = "extract"
WANDB_ENTITY_NAME = "lvjonok-korea-advanced-institute-of-science-and-technology"


class ModelTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(
            conf.exp_dir, args.path, args.prefix, args.new_dir
        )
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, "events")
        print("using log dir: ", log_dir)
        self.conf = self.postprocess_conf(conf)
        set_seeds(args.seed)

        # set up logging + training monitoring
        self.writer = self.setup_logging(conf, self.log_dir, args.run_group)
        self.setup_training_monitors()

        # buld dataset, model. logger, etc.
        train_params = AttrDict(
            logger_class=self._hp.logger,
            model_class=self._hp.model,
            n_repeat=self._hp.epoch_cycles_train,
            dataset_size=-1,
        )
        self.logger, self.model, self.train_loader = self.build_phase(
            train_params, "train"
        )

        test_params = AttrDict(
            logger_class=(
                self._hp.logger
                if self._hp.logger_test is None
                else self._hp.logger_test
            ),
            model_class=(
                self._hp.model if self._hp.model_test is None else self._hp.model_test
            ),
            n_repeat=1,
            dataset_size=args.val_data_size,
        )
        self.logger_test, self.model_test, self.val_loader = self.build_phase(
            test_params, phase="val"
        )

        # set up optimizer + evaluator
        self.optimizer = self.get_optimizer_class()(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.conf.general.use_amp)
        self.evaluator = self._hp.evaluator(
            self._hp,
            self.log_dir,
            self._hp.top_of_n_eval,
            self._hp.top_comp_metric,
            tb_logger=self.logger_test,
        )

        # load model params from checkpoint
        self.global_step, start_epoch = 0, 0
        if args.resume or conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, conf.ckpt_path)
        # self.model = torch.compile(self.model)

        # freeze the model if necessary
        if "finetune" in self.conf.data and self.conf.data.finetune:
            self.model.freeze()

        # build sampler
        if "rollout" in self._hp and self._hp.rollout:
            self.env = self._hp.environment(self.conf.env)
            # Dummy self.model_test
            self.sampler = self._hp.sampler(
                self.conf.sampler,
                self.env,
                self.model_test,
                self.logger,
                self._hp.max_rollout_len,
            )

        if args.val_sweep:
            self.run_val_sweep()
        elif args.train:
            self.train(start_epoch)
        else:
            self.val()

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "model": None,
                "model_test": None,
                "logger": None,
                "logger_test": None,
                "evaluator": None,
                "data_dir": None,  # directory where dataset is in
                "batch_size": 128,
                "exp_path": None,  # Path to the folder with experiments
                "num_epochs": 200,
                "sampler": Sampler,
                "epoch_cycles_train": 1,
                "optimizer": "radam",  # supported: 'adam', 'radam', 'rmsprop', 'sgd'
                "lr": 1e-3,
                "gradient_clip": None,
                "init_grad_clip": 0.001,
                "init_grad_clip_step": 100,  # clip gradients in initial N steps to avoid NaNs
                "momentum": 0,  # momentum in RMSProp / SGD optimizer
                "adam_beta": 0.9,  # beta1 param in Adam
                "top_of_n_eval": 1,  # number of samples used at eval time
                "top_comp_metric": None,  # metric that is used for comparison at eval time (e.g. 'mse')
                "logging_target": "wandb",
            }
        )
        return default_dict

    def generate_prior_rollouts(self, path):
        """Sample from the agent (if not loading from an RL checkpoint, this will be prior rollouts)."""
        # Remove the last two directory from the path.
        ckpt_path = path.split("/")[:-2]
        # Join with the rest of the path.
        ckpt_path = "/".join(ckpt_path)
        if "policy_params" in self.conf.agent:  # BC agent.
            self.conf.agent.policy_params.prior_model_checkpoint = ckpt_path
        else:  # SPiRL agent.
            # Close-loop SPiRL
            if (
                "policy_params" in self.conf.agent.ll_agent_params and
                "policy_model_checkpoint"
                in self.conf.agent.ll_agent_params.policy_params
            ):
                self.conf.agent.ll_agent_params.policy_params.policy_model_checkpoint = (
                    ckpt_path
                )
            else:  # Standard SPiRL
                self.conf.agent.ll_agent_params.model_checkpoint = ckpt_path
            self.conf.agent.hl_agent_params.policy_params.prior_model_checkpoint = (
                ckpt_path
            )
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)
        # Update the sampler agent.
        self.sampler._agent = self.agent
        val_rollout_storage = RolloutStorage()
        num_eval_tasks = self.env.num_eval_tasks
        if num_eval_tasks is None:
            num_eval_tasks = self.args.n_val_samples
        with self.agent.val_mode():
            with torch.no_grad():
                with timing("Eval rollout time: "):
                    for i in range(num_eval_tasks):
                        val_rollout_storage.append(
                            self.sampler.sample_episode(
                                is_train=False, render=True, env_id=i
                            )
                        )
        rollout_stats = val_rollout_storage.rollout_stats()
        with timing("Eval log time: "):
            self.agent.log_outputs(
                rollout_stats,
                val_rollout_storage,
                self.logger,
                log_images=True,
                step=self.global_step,
            )
            print("Evaluation Avg_Reward: {}".format(rollout_stats.avg_reward))
        del val_rollout_storage

    def train(self, start_epoch):
        if not self.args.skip_first_val:
            self.val()

        for epoch in range(start_epoch, self._hp.num_epochs):
            self.train_epoch(epoch)

            if not self.args.dont_save:
                if epoch % 5 == 0:
                    path = save_checkpoint(
                        {
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "grad_scaler": self.grad_scaler.state_dict(),
                        },
                        os.path.join(self._hp.exp_path, "weights"),
                        CheckpointHandler.get_ckpt_name(epoch),
                    )
                    if "rollout" in self._hp and self._hp.rollout:
                        self.generate_prior_rollouts(path)

            if epoch % self.args.val_interval == 0:
                self.val()

    def train_epoch(self, epoch):
        self.model.train()
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)

        print("starting epoch ", epoch)

        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            with self.training_context():
                with torch.cuda.amp.autocast(enabled=self.conf.general.use_amp):
                    self.optimizer.zero_grad()
                    output = self.model(inputs)
                    losses = self.model.loss(output, inputs)
                    self.grad_scaler.scale(losses.total.value).backward()
                    self.call_hooks(inputs, output, losses, epoch)

                    if self.global_step < self._hp.init_grad_clip_step:
                        # clip gradients in initial steps to avoid NaN gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self._hp.init_grad_clip
                        )
                    self.grad_scaler.step(self.optimizer)
                    self.model.step()
                    self.grad_scaler.update()

            if self.args.train_loop_pdb:
                import pdb

                pdb.set_trace()

            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.args.dont_save:
                self.model.log_outputs(
                    output,
                    inputs,
                    losses,
                    self.global_step,
                    log_images=self.log_images_now,
                    phase="train",
                    **self._logging_kwargs,
                )
            batch_time.update(time.time() - end)
            end = time.time()

            if self.log_outputs_now:
                print(
                    "GPU {}: {}".format(
                        os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else "none",
                        self._hp.exp_path,
                    )
                )
                print(
                    (
                        "itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            self.global_step,
                            epoch,
                            self.batch_idx,
                            len(self.train_loader),
                            100.0 * self.batch_idx / len(self.train_loader),
                            losses.total.value.item(),
                        )
                    )
                )

                print(
                    "avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s".format(
                        data_load_time.avg,
                        batch_time.avg - upto_log_time.avg,
                        upto_log_time.avg - data_load_time.avg,
                        batch_time.avg,
                    )
                )
                togo_train_time = (
                    batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.0
                )
                print("ETA: {:.2f}h".format(togo_train_time))

            del output, losses
            self.global_step = self.global_step + 1

    def val(self):
        if "finetune" in self.conf.data and self.conf.data.finetune:
            # no need to validate when doing offline finetuning
            return
        print("Running Testing")
        if self.args.test_prediction:
            start = time.time()
            self.model_test.load_state_dict(self.model.state_dict())
            losses_meter = RecursiveAverageMeter()
            self.model_test.eval()
            self.evaluator.reset()
            with autograd.no_grad():
                for batch_idx, sample_batched in enumerate(self.val_loader):
                    inputs = AttrDict(
                        map_dict(lambda x: x.to(self.device), sample_batched)
                    )

                    # run evaluator with val-mode model
                    with self.model_test.val_mode():
                        self.evaluator.eval(inputs, self.model_test)

                    # run non-val-mode model (inference) to check overfitting
                    output = self.model_test(inputs)
                    losses = self.model_test.loss(output, inputs)

                    losses_meter.update(losses)
                    del losses

                if not self.args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)

                    self.model_test.log_outputs(
                        output,
                        inputs,
                        losses_meter.avg,
                        self.global_step,
                        log_images=True,
                        phase="val",
                        **self._logging_kwargs,
                    )
                    print(
                        (
                            "\nTest set: Average loss: {:.4f} in {:.2f}s\n".format(
                                losses_meter.avg.total.value.item(), time.time() - start
                            )
                        )
                    )
            del output

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device("cuda") if self.use_cuda else torch.device("cpu")
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print("loading from the config file {}".format(conf.conf_path))
        conf_module = imp.load_source("conf", conf.conf_path)
        conf.general = conf_module.configuration
        conf.model = conf_module.model_config

        # data config
        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source(
                "dataset_spec", os.path.join(AttrDict(conf).data_dir, "dataset_spec.py")
            )
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
            data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
        conf.data = data_conf

        if "finetune" in conf.data and conf.data.finetune:
            conf.agent = conf_module.agent_config
            conf.agent.device = self.device
            conf.sampler = (
                conf_module.sampler_config
                if hasattr(conf_module, "sampler_config")
                else AttrDict({})
            )
            # model loading config
            conf.ckpt_path = (
                conf.agent.checkpt_path if "checkpt_path" in conf.agent else None
            )
            if "env_config" in conf_module.__dict__:
                conf.env = conf_module.env_config
                conf.env.gpu = self.args.gpu

        # model loading config
        conf.ckpt_path = (
            conf.model.checkpt_path if "checkpt_path" in conf.model else None
        )

        # load config overwrites
        if self.args.config_override != "":
            for override in self.args.config_override.split(","):
                key_str, value_str = override.split("=")
                # special bool handling
                if value_str.lower() == "true":
                    value_str = True
                elif value_str.lower() == "false":
                    value_str = False
                keys = key_str.split(".")
                curr = conf
                for key in keys[:-1]:
                    curr = curr[key]
                print(f"Overwriting {keys[-1]} from {curr[keys[-1]]} to {value_str}")
                if type(curr[keys[-1]]) != type(None):
                    curr[keys[-1]] = type(curr[keys[-1]])(value_str)
                else:
                    curr[keys[-1]] = float(value_str)
        return conf

    def postprocess_conf(self, conf):
        conf.model["batch_size"] = (
            self._hp.batch_size
            if not torch.cuda.is_available()
            else int(self._hp.batch_size / torch.cuda.device_count())
        )
        conf.model.update(conf.data.dataset_spec)
        conf.model["device"] = conf.data["device"] = self.device.type
        return conf

    def setup_logging(self, conf, log_dir, run_name):
        if not self.args.dont_save:
            print("Writing to the experiment directory: {}".format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(
                conf.conf_path,
                os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"),
            )
            if self._hp.logging_target == "wandb":
                writer = WandBLogger(
                    self.args.run_name,
                    self.args.run_group,
                    WANDB_PROJECT_NAME,
                    entity=WANDB_ENTITY_NAME,
                    path=self._hp.exp_path,
                    conf=conf,
                    exclude=["model_rewards", "data_dataset_spec_rewards"],
                )
            else:
                writer = SummaryWriter(log_dir)
        else:
            writer = None

        # set up additional logging args
        self._logging_kwargs = AttrDict()
        return writer

    def setup_training_monitors(self):
        self.training_context = (
            autograd.detect_anomaly if self.args.detect_anomaly else dummy_context
        )
        self.hooks = []
        self.hooks.append(LossSpikeHook("sg_img_mse_train"))
        self.hooks.append(NanGradHook(self))
        self.hooks.append(NoneGradHook(self))

    def build_phase(self, params, phase):
        if not self.args.dont_save:
            if self._hp.logging_target == "wandb":
                logger = self.writer
            else:
                logger = params.logger_class(self.log_dir, summary_writer=self.writer)
        else:
            logger = None
        model = params.model_class(self.conf.model, logger)
        if torch.cuda.device_count() > 1:
            raise ValueError(
                "Detected {} devices. Currently only single-GPU training is supported!".format(
                    torch.cuda.device_count()
                ),
                "Set CUDA_VISIBLE_DEVICES=<desired_gpu_id>.",
            )
            # print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            # model = DataParallelWrapper(model)
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(
            self.args,
            model,
            self.conf.data,
            phase,
            params.n_repeat,
            params.dataset_size,
        )
        return logger, model, loader

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        if args.feed_random_data:
            dataset_class = RandomVideoDataset
        else:
            dataset_class = data_conf.dataset_spec.dataset_class

        loader = dataset_class(
            self._hp.data_dir,
            data_conf,
            resolution=model.resolution,
            phase=phase,
            shuffle=phase == "train",
            dataset_size=dataset_size,
        ).get_data_loader(self._hp.batch_size, n_repeat)

        return loader

    def resume(self, ckpt, path=None):
        path = (
            os.path.join(self._hp.exp_path, "weights")
            if path is None
            else os.path.join(path, "weights")
        )
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        self.global_step, start_epoch, _ = CheckpointHandler.load_weights(
            weights_file,
            self.model,
            load_step=True,
            load_opt=True,
            optimizer=self.optimizer,
            grad_scaler=self.grad_scaler,
            strict=self.args.strict_weight_loading,
        )
        self.model.to(self.model.device)
        return start_epoch

    def get_optimizer_class(self):
        optim = self._hp.optimizer
        if optim == "adam":
            get_optim = partial(
                get_clipped_optimizer,
                optimizer_type=Adam,
                betas=(self._hp.adam_beta, 0.999),
            )
        elif optim == "adamw":
            get_optim = partial(
                get_clipped_optimizer,
                optimizer_type=AdamW,
                betas=(self._hp.adam_beta, 0.999),
            )
        elif optim == "radam":
            get_optim = partial(
                get_clipped_optimizer,
                optimizer_type=RAdam,
                betas=(self._hp.adam_beta, 0.999),
            )
        elif optim == "rmsprop":
            get_optim = partial(
                get_clipped_optimizer,
                optimizer_type=RMSprop,
                momentum=self._hp.momentum,
            )
        elif optim == "sgd":
            get_optim = partial(
                get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum
            )
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optim))
        return partial(get_optim, gradient_clip=self._hp.gradient_clip)

    def run_val_sweep(self):
        epochs = CheckpointHandler.get_epochs(
            os.path.join(self._hp.exp_path, "weights")
        )
        for epoch in list(sorted(epochs))[::2]:
            self.resume(epoch)
            self.val()
        return

    def get_exp_dir(self):
        return os.environ["EXP_DIR"]

    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0

    @property
    def log_outputs_now(self):
        return (
            self.global_step % self.log_outputs_interval == 0
            or self.global_step % self.log_images_interval == 0
        )


def save_checkpoint(state, folder, filename="checkpoint.pth"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    torch.save(state, path)
    print(f"Saved checkpoint to {path}!")
    return path


def get_exp_dir():
    return os.environ["EXP_DIR"]


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split("configs/", 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)


if __name__ == "__main__":
    ModelTrainer(args=get_args())
