import itertools
import logging
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from transformers import Trainer, TrainerCallback
from transformers.utils import is_sagemaker_mp_enabled, ExplicitEnum

from .sft import SFT
from .sft_args import SftArguments
from .reg_args import RegArguments

logger = logging.getLogger(__name__)

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

class ShardedDDPOption(ExplicitEnum):
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


class _RegLossCalculationCallback(TrainerCallback):

    def __init__(self, sft):
        self._sft = sft

    def on_step_begin(self, args, state, control, **kwargs):
        self._sft.calculate_reg_loss = True


class SparseFineTuner(Trainer):
    """ Superclass for Trainers that learn sparse fine-tunings. Keeps track
    of original model parameters so that difference vectors can be calculated
    at the end of training, and which parameters are masked so that gradients
    of fixed parameters can be zeroed.

    Args:
        sft_args: an SftArguments object containing SFT training options.
        maskable_params: a list of parameter names; the model parameters which
            are to be sparsely fine-tuned. Parameters not included in
            maskable_params but have requires_grad=True will be fully
            fine-tuned (this is typically preferable for model heads, for
            instance). If None, all parameters will be sparsely fine-tuned.
        **kwargs: arguments to pass to Trainer constructor.
    """
    def __init__(
        self,
        *args,
        sft_args=None,
        reg_args=None,
        maskable_params=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        logger.setLevel(self.args.get_process_log_level())

        if sft_args is None:
            self.sft_args = SftArguments()
        else:
            self.sft_args = sft_args

        if reg_args is None:
            self.reg_args = RegArguments()
        else:
            self.reg_args = reg_args

        if maskable_params is None:
            self.maskable_params = set(
                n for n, _ in self.model.named_parameters()
            )
        else:
            self.maskable_params = set(maskable_params)

        self._num_params = sum(
            p.data.numel()
            for n, p in self.model.named_parameters()
        )
        self._num_maskable_params = sum(
            p.data.numel()
            for n, p in self.model.named_parameters()
            if n in self.maskable_params
        )

        self._regularized = (
            self.sft_args.full_l1_reg != 0.0 or
            self.sft_args.sparse_l1_reg != 0.0
        )
        # Since the regularization loss is dependent only on the parameter
        # values, we can get away with calculating it only once per full step
        # rather than at every gradient accumulation step. This flag gets set
        # by a _RegLossCalculationCallback at the start of each full step to
        # tell us to do so.
        self.calculate_reg_loss = False
        self._reg_loss = 0.0 # Keeps track of the reg loss for logging purposes.
        if self._regularized:
            # If regularization is in use, the original parameters should be
            # kept on the same device as the tuned parameters for efficiency.
            device = None
            self.add_callback(_RegLossCalculationCallback(self))
        else:
            # Otherwise we can save some GPU RAM by keeping them on the CPU.
            device = 'cpu'
            
        self._original_params = {
            n: torch.zeros_like(p, device=device).copy_(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self._mask = {
            n: torch.ones_like(p, dtype=torch.bool)
            for n, p in self.model.named_parameters()
            if n in self.maskable_params
        }
        # Whether to apply masking during training.
        self._masking_enabled = True

    def enable_masking(self):
        self._masking_enabled = True

    def disable_masking(self):
        self._masking_enabled = False

    def reset(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self._original_params[n])

    def freeze(self):
        for _, p in self._mask.items():
            p.data.zero_()

    def sft(self, eps=1e-7):
        """ Calculates the sparse difference vector between the current
        parameter values and the pre-trained values.

        Args:
            eps: differences smaller than this amount will be treated as zero,
            i.e. excluded from the SFT.

        Returns:
            An SFT containing the differences.
        """
        with torch.no_grad():
            diffs = SFT()
            for n, p in self.model.named_parameters():
                if n in self.maskable_params:
                    delta = p - self._original_params[n].to(p.device)
                    abs_delta = torch.abs(delta)
                    significant = abs_delta > eps
                    delta = delta * significant
                    diffs.add_param(n, delta, diff=True)
                elif p.requires_grad:
                    # p is to be stored in full rather than as a difference.
                    # Typically this happens when p belongs to the model head.
                    diffs.add_param(n, p, diff=False)
            return diffs

    def set_training_len(self, min_steps, max_steps, max_epochs):
        if max_steps is None and max_epochs is None:
            raise ValueError('Length of sft training not specified.')
        if min_steps is not None and max_steps is not None and min_steps > max_steps:
            raise ValueError('min_steps cannot be > max_steps')

        if max_epochs is None:
            self.args.max_steps = max_steps
        else:
            n_steps = max_epochs * len(self.train_dataset) // (
                self.args.per_device_train_batch_size *
                self.args.gradient_accumulation_steps
            )
            logger.info(f'{max_epochs} epochs = {n_steps} steps')
        
            if max_steps is None or n_steps < max_steps:
                if min_steps is not None and n_steps < min_steps:
                    self.args.max_steps = min_steps
                else:
                    self.args.num_train_epochs = max_epochs
                    self.args.max_steps = -1
            else:
                self.args.max_steps = max_steps

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        l1_reg = (
            self.sft_args.sparse_l1_reg
            if self._masking_enabled
            else self.sft_args.full_l1_reg
        )
        if l1_reg != 0.0 and self.calculate_reg_loss:
            # Since we only calculate reg loss once per full step.
            l1_reg *= self.args.gradient_accumulation_steps
            l1_dists = []
            for n, p in self.model.named_parameters():
                if (
                    p.requires_grad and
                    (n in self.maskable_params or
                        not self.sft_args.apply_reg_to_sparse_only)
                ):
                    l1_dists.append(
                        torch.sum(torch.abs(p - self._original_params[n]))
                    )
            reg_loss = l1_reg * torch.sum(torch.stack(l1_dists)) / self._num_params
            reg_loss.backward()
            self._reg_loss += float(reg_loss)
            self.calculate_reg_loss = False

        if self._masking_enabled:
            # set gradients for non-trainable parametres to zero.
            for n, p in self.model.named_parameters():
                if n in self.maskable_params and p.grad is not None:
                    p.grad *= self._mask[n]
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self._reg_loss != 0.0:
                logs['l1_reg_loss'] = round(self._reg_loss / (self.state.global_step - self._globalstep_last_logged), 4)
                self._reg_loss = 0.0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        adv_parameters =[(n,p) for n,p in self.model.named_parameters() if any(x in n for x in ['adv_'])]
        adv_parameters_names = [n for n,p in self.model.named_parameters() if any(x in n for x in ['adv_'])]

        if adv_parameters:
            logger.info('Parameters for adversarial learning {}'.format([x[0] for x in adv_parameters]))

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in adv_parameters_names)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in adv_parameters_names)
                    ],
                    "weight_decay": 0.0,
                },
            ]
                        
            if adv_parameters:
                optimizer_grouped_parameters.append({
                    'params': [p for n, p in adv_parameters],
                    'initial_lr': self.args.learning_rate * self.reg_args.adv_lr_scale
                })

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

