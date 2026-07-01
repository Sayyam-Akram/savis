try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 4080 single GPU (changed from '0,1')

import copy
import itertools
import logging

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from avism import (
    AVISDatasetMapper,
    AVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    add_avism_config,
)


def get_num_gt_instances(batched_inputs):
    total_instances = 0
    for video in batched_inputs:
        instances = video.get("instances", [])
        if not instances or len(instances) == 0:
            continue
        num_insts = len(instances[0])
        if num_insts == 0:
            continue
        if not hasattr(instances[0], "gt_ids"):
            continue
        gt_ids_list = [inst.gt_ids for inst in instances]
        gt_ids = torch.stack(gt_ids_list, dim=1) # [num_insts, num_frames]
        valid_bool = (gt_ids != -1).any(dim=1)
        total_instances += valid_bool.sum().item()
    return total_instances


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return AVISEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AVISDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        mapper = AVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def __init__(self, cfg):
        super().__init__(cfg)
        self.gradient_accumulation_steps = cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)
        
    def run_step(self):
        """
        Implement the standard training step logic with gradient accumulation and AMP.
        """
        if self.gradient_accumulation_steps <= 1:
            super().run_step()
            return
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[SimpleTrainer] CUDA is required for the trainer!"
        
        from detectron2.utils.events import get_event_storage
        from torch.cuda.amp import autocast
        import time

        start = time.perf_counter()
        
        # Ensure data loader iterator exists (D2 sometimes initializes it lazily or in train())
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)

        # 1. Pre-fetch all data batches for the accumulation cycle
        batches = []
        for accum_step in range(self.gradient_accumulation_steps):
            try:
                batches.append(next(self._data_loader_iter))
            except StopIteration:
                self._data_loader_iter = iter(self.data_loader)
                batches.append(next(self._data_loader_iter))

        # 2. Count the ground truth instances for each batch, and get the global total
        local_gts = []
        for batch in batches:
            local_gts.append(max(get_num_gt_instances(batch), 1))
        num_gt_total = sum(local_gts)

        total_loss_dict = {}
        
        # Zero gradients only at the start of accumulation cycle
        self.optimizer.zero_grad()
        
        for accum_step in range(self.gradient_accumulation_steps):
            data = batches[accum_step]
            data_time = time.perf_counter() - start

            # Run forward pass with autocast
            with autocast(enabled=self.grad_scaler.is_enabled()):
                loss_dict = self.model(data)

            if isinstance(loss_dict, torch.Tensor):
                # If loss_dict is a single tensor, we scale it by steps
                losses = loss_dict / self.gradient_accumulation_steps
                loss_dict = {"total_loss": loss_dict}
            else:
                # Scaled losses for instance/mask losses vs classification/other losses
                scaled_loss_terms = []
                for k, v in loss_dict.items():
                    if "mask" in k or "dice" in k:
                        # Normalize mask/dice loss dynamically using N_local and N_total
                        scaled_val = v * (local_gts[accum_step] / num_gt_total)
                    else:
                        # Standard scaling for other losses
                        scaled_val = v / self.gradient_accumulation_steps
                    scaled_loss_terms.append(scaled_val)
                losses = sum(scaled_loss_terms)
            
            # Check for invalid loss
            if not torch.isfinite(losses).all():
                raise FloatingPointError(
                    "Loss became infinite or NaN in accumulation step {}/{}!".format(
                        accum_step + 1, self.gradient_accumulation_steps
                    )
                )

            # Backward pass with GradScaler
            self.grad_scaler.scale(losses).backward()
            
            # Aggregate metrics for logging
            with torch.no_grad():
                for k, v in loss_dict.items():
                    if k not in total_loss_dict:
                        total_loss_dict[k] = v.item() / self.gradient_accumulation_steps
                    else:
                        total_loss_dict[k] += v.item() / self.gradient_accumulation_steps
                
                # Track total loss
                if "total_loss" not in total_loss_dict:
                     total_loss_dict["total_loss"] = losses.item()
                else:
                     total_loss_dict["total_loss"] += losses.item()

            start = time.perf_counter() # Reset start time for next data loading

        # Optimizer step only after accumulation is done, scaled
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        # Logging
        storage = get_event_storage()
        storage.put_scalars(data_time=data_time, **total_loss_dict)


    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        if cfg["eval_only"]:
            from torch.cuda.amp import autocast
            logger = logging.getLogger(__name__)
            if isinstance(evaluators, DatasetEvaluator):
                evaluators = [evaluators]
            if evaluators is not None:
                assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                    len(cfg.DATASETS.TEST), len(evaluators)
                )

            results = OrderedDict()
            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                data_loader = cls.build_test_loader(cfg, dataset_name)
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = cls.build_evaluator(cfg, dataset_name)
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                with autocast():
                    results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i

                print("AP: {} || AP_s: {} || AP_m: {} || AP_l: {} || AR: {}".format(results_i['segm']['AP_all'],
                                                                                    results_i['segm']['AP_s'],
                                                                                    results_i['segm']['AP_m'],
                                                                                    results_i['segm']['AP_l'],
                                                                                    results_i['segm']['AR_all']))

                print("DetA: {} || DetRe: {} || DetPr: {}".format(results_i['segm']['DetA'],
                                                                  results_i['segm']['DetRe'],
                                                                  results_i['segm']['DetPr']))

                print("AssA: {} || AssRe: {} || AssPr: {}".format(results_i['segm']['AssA'],
                                                                  results_i['segm']['AssRe'],
                                                                  results_i['segm']['AssPr']))

                print("HOTA: {} || LocA: {} || DetA: {} || AssA: {}".format(results_i['segm']['HOTA'],
                                                                            results_i['segm']['LocA'],
                                                                            results_i['segm']['DetA'],
                                                                            results_i['segm']['AssA']))

                print("FSLAn_count: {} || FSLAn_all: {} || FSLAs_count: {} || FSLAs_all: {} || FSLAm_count: {} || FSLAm_all: {}".format(
                    results_i['segm']['FAn_count'],
                    results_i['segm']['FAn_all'],
                    results_i['segm']['FAs_count'],
                    results_i['segm']['FAs_all'],
                    results_i['segm']['FAm_count'],
                    results_i['segm']['FAm_all']))

                print("FSLA: {} || FSLAn: {} || FSLAs: {} || FSLAm: {}".format(results_i['segm']['FA'],
                                                                               results_i['segm']['FAn'],
                                                                               results_i['segm']['FAs'],
                                                                               results_i['segm']['FAm']))

                if 'J&F' in results_i['segm']:
                    print("J&F: {} || J-Mean: {} || J-Recall: {} || F-Mean: {} || F-Recall: {}".format(
                        results_i['segm']['J&F'],
                        results_i['segm']['J-Mean'],
                        results_i['segm']['J-Recall'],
                        results_i['segm']['F-Mean'],
                        results_i['segm']['F-Recall']))

            if len(results) == 1:
                results = list(results.values())[0]
            return results
        else:
            pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_avism_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg["eval_only"] = args.eval_only
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "avism" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="avism")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
