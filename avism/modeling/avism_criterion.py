import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class AvismSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, sim_use_clip,
                 agcl_temperature=0.07, calib_hard_weight=3.0):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.sim_use_clip = sim_use_clip

        # AGCL: Audio-Guided Contrastive Learning (SeaVIS-inspired)
        self.agcl_temperature = agcl_temperature
        hidden_dim = 256  # Must match HIDDEN_DIM
        self.audio_anchor_proj = nn.Linear(hidden_dim, hidden_dim)

        self.calib_hard_weight = calib_hard_weight

    def loss_frame_contrastive(self, outputs, clip_targets, frame_targets,
                                clip_indices, frame_indices, num_masks):
        """
        Frame-Level Audio-Guided Contrastive Loss (SeaVIS Eq. 3-4).
        """
        src_fq = outputs.get("pred_fq_embed")  # [L, B, T, fQ, C]
        if src_fq is None:
            return {"loss_agcl_frame": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
        L, B, T, fQ, C = src_fq.shape
        BT = B * T
        
        # Get audio features (already projected to hidden_dim by the decoders)
        audio_feats = outputs.get("audio_feats_proj")  # [B*T, C] or [B, T, C]
        if audio_feats is None:
            return {"loss_agcl_frame": torch.tensor(0.0, device=src_fq.device)}
        
        # Project audio to anchor space and normalize
        audio_anchors = self.audio_anchor_proj(audio_feats)  # [B*T, C]
        if audio_anchors.dim() == 2:
            audio_anchors = audio_anchors.view(B, T, C)
        audio_anchors = F.normalize(audio_anchors, dim=-1)   # [B, T, C]
        
        # Use only the last decoder layer's embeddings (most refined)
        fq = src_fq[-1]  # [B, T, fQ, C]
        
        # Flatten frame indices (they come grouped by decoder layer)
        flat_frame_indices = frame_indices[-1] if isinstance(frame_indices[0], list) else frame_indices
        
        # Flatten B & T dimensions for vectorization
        fq_flat = fq.view(BT, fQ, C)  # [BT, fQ, C]
        fq_norm = F.normalize(fq_flat, dim=-1)  # [BT, fQ, C]
        
        audio_anchors_flat = audio_anchors.view(BT, C)  # [BT, C]
        
        # Compute all similarities in one GPU batch matrix multiplication
        # [BT, fQ, C] @ [BT, C, 1] -> [BT, fQ, 1] -> [BT, fQ]
        all_sims = torch.bmm(fq_norm, audio_anchors_flat.unsqueeze(-1)).squeeze(-1) / self.agcl_temperature
        
        # Build positive mask
        pos_mask = torch.zeros((BT, fQ), dtype=torch.bool, device=fq.device)
        for i in range(BT):
            if i < len(flat_frame_indices):
                matched_src, _ = flat_frame_indices[i]
                if len(matched_src) > 0:
                    pos_mask[i, matched_src] = True
        
        num_valid_pairs = pos_mask.sum().item()
        if num_valid_pairs == 0:
            return {"loss_agcl_frame": torch.tensor(0.0, device=fq.device)}
            
        # Numerical stability: subtract max per frame
        max_sims, _ = all_sims.max(dim=-1, keepdim=True)  # [BT, 1]
        exp_sims = torch.exp(all_sims - max_sims)  # [BT, fQ]
        
        sum_all_exp = exp_sims.sum(dim=-1, keepdim=True)  # [BT, 1]
        sum_pos_exp = (exp_sims * pos_mask).sum(dim=-1, keepdim=True)  # [BT, 1]
        sum_neg_exp = sum_all_exp - sum_pos_exp  # [BT, 1]
        
        # Log-Sum-Exp denominator for each positive entry
        denom = exp_sims + sum_neg_exp  # [BT, fQ]
        loss_all = torch.log(denom) - (all_sims - max_sims)  # [BT, fQ]
        
        # Compute InfoNCE only over positive entries
        total_loss = loss_all[pos_mask].sum() / num_valid_pairs
        
        return {"loss_agcl_frame": total_loss}

    def loss_instance_contrastive(self, outputs, clip_targets, frame_targets,
                                   clip_indices, frame_indices, num_masks):
        """
        Instance-Level Audio-Guided Contrastive Loss (SeaVIS Eq. 5-6).
        """
        src_cq = outputs.get("pred_cq_embed")  # [L, B, cQ, C]
        audio_feats = outputs.get("audio_feats_proj")  # [B*T, C] or [B, T, C]
        
        if src_cq is None or audio_feats is None:
            dev = next(iter(outputs.values())).device
            return {"loss_agcl_instance": torch.tensor(0.0, device=dev)}
        
        L, B, cQ, C = src_cq.shape
        cq = src_cq[-1]  # [B, cQ, C] - last decoder layer
        
        if audio_feats.dim() == 2:
            BT = audio_feats.shape[0]
            T = BT // B
            audio_feats = audio_feats.view(B, T, C)
        else:
            T = audio_feats.shape[1]
            
        audio_anchors = self.audio_anchor_proj(
            audio_feats.reshape(B * T, C)
        ).view(B, T, C)  # [B, T, C]
        
        losses = []
        
        for b in range(B):
            matched_src, matched_tgt = clip_indices[b]
            if len(matched_src) == 0:
                continue
                
            # Pre-normalize clip queries for this batch element
            cq_b = F.normalize(cq[b, matched_src], dim=-1)  # [M, C]
            
            # Target IDs for matched instances
            ids_all = clip_targets[b]["ids"][matched_tgt]  # [M, T]
            
            sounding_mask = ids_all != -1  # [M, T]
            silent_mask = ids_all == -1    # [M, T]
            
            # Loop over matched instances in this batch element
            for i, (sounding, silent) in enumerate(zip(sounding_mask, silent_mask)):
                if not sounding.any() or not silent.any():
                    continue
                    
                # Sounding audio anchor
                sounding_audio = audio_anchors[b, sounding]  # [N_sound, C]
                anchor = F.normalize(sounding_audio.mean(0), dim=-1)  # [C]
                
                pos_sim = (cq_b[i] @ anchor) / self.agcl_temperature
                
                # Negatives: audio from silent frames
                silent_audio = audio_anchors[b, silent]  # [N_silent, C]
                silent_audio_norm = F.normalize(silent_audio, dim=-1)
                neg_sims = (silent_audio_norm @ cq_b[i]) / self.agcl_temperature  # [N_silent]
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                target = torch.zeros(1, dtype=torch.long, device=logits.device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                
        if len(losses) > 0:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=cq.device)
            
        return {"loss_agcl_instance": total_loss}

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        L, B, cQ, _ = src_logits.shape
        src_logits = src_logits.reshape(L*B, cQ, self.num_classes+1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets * L, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_avism_ce': loss_ce}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        if "pred_masks" not in outputs and "mask_features" in outputs:
            outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", outputs["pred_mask_embed"], outputs["mask_features"])
        assert "pred_masks" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        L, B, cQ, T, H, W = src_masks.shape
        src_masks = src_masks.reshape(L*B, cQ, T, H, W)

        src_masks = src_masks[idx] # Nt x T x Hp x Wp
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets * L, indices)]).to(src_masks)
        # Nt x T x Ht x Wt
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # Nt*T, randN -> Nt, T*randN
        point_logits = point_logits.view(len(idx[0]), T * self.num_points)
        point_labels = point_labels.view(len(idx[0]), T * self.num_points)

        losses = {
            "loss_avism_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_avism_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        if "mask_features" in outputs and "pred_masks" in outputs:
            del outputs["pred_masks"]
        return losses

    def loss_fg_sim(
        self, outputs, clip_targets, frame_targets,
        clip_indices, frame_indices, num_masks, MULTIPLIER=1000
    ):
        total_src_q, total_tgt_ids, total_batch_idx = [], [], []

        # Frame
        src_fq = outputs["pred_fq_embed"]   # L, B, T, fQ, C
        # L = number of frame_decoder layers
        L, B, T, fQ, C = src_fq.shape
        src_fq = src_fq.flatten(0, 2)       # LBT, fQ, C

        frame_indices = sum(frame_indices, [])
        frame_src_idx = self._get_src_permutation_idx(frame_indices)    # len = LBT
        src_fq = src_fq[frame_src_idx]      # Nf, C
        target_frame_ids = torch.cat(
            [t["ids"][J] for t, (_, J) in zip(frame_targets * L, frame_indices)]
        )
        frame_batch_idx = torch.div(frame_src_idx[0].to(device=src_fq.device), T, rounding_mode="floor")
        is_frame_valid = target_frame_ids != -1
        target_frame_ids += frame_batch_idx * MULTIPLIER

        total_src_q.append(src_fq[is_frame_valid])
        total_tgt_ids.append(target_frame_ids[is_frame_valid])
        total_batch_idx.append(frame_batch_idx[is_frame_valid])

        # Clip
        if self.sim_use_clip:
            src_cq = outputs["pred_cq_embed"]   # L, B, cQ, C
            src_cq = src_cq.flatten(0, 1)       # LB , cQ, C

            clip_src_idx = self._get_src_permutation_idx(clip_indices)      # len = LB
            src_cq = src_cq[clip_src_idx]       # Nc, C
            target_clip_ids = torch.cat(        # clip_ids' shape = (N, num_frames) -> (N,)
                [t["ids"][J] for t, (_, J) in zip(clip_targets * L, clip_indices)]
            ).amax(dim=1)
            clip_batch_idx = clip_src_idx[0].to(device=src_fq.device)
            is_clip_valid = target_clip_ids != -1
            target_clip_ids += clip_batch_idx * MULTIPLIER

            total_src_q.append(src_cq[is_clip_valid])
            total_tgt_ids.append(target_clip_ids[is_clip_valid])
            total_batch_idx.append(clip_batch_idx[is_clip_valid])

        # Clip + Frame
        total_src_q = torch.cat(total_src_q)            # Nc+Nf, C
        total_tgt_ids = torch.cat(total_tgt_ids)        # Nc+Nf
        total_batch_idx = torch.cat(total_batch_idx)    # Nc+Nf

        sim_pred_logits = torch.matmul(total_src_q, total_src_q.T)          # Nc+Nf, Nc+Nf
        sim_tgt = (total_tgt_ids[:, None] == total_tgt_ids[None]).float()   # Nc+Nf, Nc+Nf

        same_clip = (total_batch_idx[:, None] == total_batch_idx[None]).float()
        loss = F.binary_cross_entropy_with_logits(sim_pred_logits, sim_tgt, reduction='none')

        loss = loss * same_clip
        loss_clip_sim = loss.sum() / (same_clip.sum() + 1e-6)

        return {"loss_clip_sim": loss_clip_sim}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_calibration(self, outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks):
        device = next(iter(outputs.values())).device if isinstance(outputs, dict) else next(iter(outputs)).device
        loss_calib = None

        # --- Stage 1 Frame-level Calibration Loss ---
        stage1_outputs = outputs.get("stage1_outputs") if isinstance(outputs, dict) else None
        if stage1_outputs is not None and "pred_calib_logits" in stage1_outputs:
            r_final = stage1_outputs["pred_calib_logits"] # [BT, fQ, 1]
            BT, fQ, _ = r_final.shape
            device = r_final.device

            if frame_indices is not None:
                y = torch.zeros((BT, fQ), dtype=torch.float32, device=device)
                flat_frame_indices = frame_indices[-1] if isinstance(frame_indices[0], list) else frame_indices
                for i in range(BT):
                    if i < len(flat_frame_indices):
                        matched_src, _ = flat_frame_indices[i]
                        if len(matched_src) > 0:
                            y[i, matched_src] = 1.0

                r_final_sq = r_final.squeeze(-1)
                pos_weight = torch.tensor([15.0]).to(device)
                loss_calib = F.binary_cross_entropy_with_logits(
                    r_final_sq, y, pos_weight=pos_weight.expand_as(y), reduction="mean"
                )

                if "aux_outputs" in stage1_outputs:
                    for aux_out in stage1_outputs["aux_outputs"]:
                        if "pred_calib_logits" in aux_out:
                            r_aux = aux_out["pred_calib_logits"].squeeze(-1)
                            loss_calib += F.binary_cross_entropy_with_logits(
                                r_aux, y, pos_weight=pos_weight.expand_as(y), reduction="mean"
                            )

        # --- Stage 2 Clip-level Calibration Loss ---
        if isinstance(outputs, dict) and "pred_calib_logits" in outputs:
            r_final_stage2 = outputs["pred_calib_logits"] # [L, B, cQ, 1]
            L, B, cQ, _ = r_final_stage2.shape
            device = r_final_stage2.device

            r_final_stage2 = r_final_stage2.reshape(L * B, cQ, 1)

            y_stage2 = torch.zeros((B, cQ), dtype=torch.float32, device=device)
            if clip_indices is not None:
                for b in range(B):
                    if b < len(clip_indices):
                        matched_src, _ = clip_indices[b]
                        if len(matched_src) > 0:
                            y_stage2[b, matched_src] = 1.0

            # Duplicate y_stage2 L times to match L * B batch size
            y_stage2 = y_stage2.unsqueeze(0).repeat(L, 1, 1).reshape(L * B, cQ)

            r_final_sq_stage2 = r_final_stage2.squeeze(-1)
            pos_weight = torch.tensor([15.0]).to(device)
            loss_calib_stage2 = F.binary_cross_entropy_with_logits(
                r_final_sq_stage2, y_stage2, pos_weight=pos_weight.expand_as(y_stage2), reduction="mean"
            )

            if "aux_outputs" in outputs:
                for aux_out in outputs["aux_outputs"]:
                    if "pred_calib_logits" in aux_out:
                        r_aux = aux_out["pred_calib_logits"] # [L, B, cQ, 1]
                        r_aux = r_aux.reshape(L * B, cQ, 1).squeeze(-1)
                        loss_calib_stage2 += F.binary_cross_entropy_with_logits(
                            r_aux, y_stage2, pos_weight=pos_weight.expand_as(y_stage2), reduction="mean"
                        )
            
            if loss_calib is None:
                loss_calib = loss_calib_stage2
            else:
                loss_calib += loss_calib_stage2

        if loss_calib is None:
            return {"loss_calib": torch.tensor(0.0, device=device)}

        return {"loss_calib": loss_calib}


    def get_loss(
        self, loss, outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
    ):
        loss_map = {
            'avism_labels': self.loss_labels,
            'avism_masks': self.loss_masks,
            'fg_sim': self.loss_fg_sim,
            'agcl_frame': self.loss_frame_contrastive,
            'agcl_instance': self.loss_instance_contrastive,
            'calib': self.loss_calibration,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ('fg_sim', 'agcl_frame', 'agcl_instance', 'calib'):
            return loss_map[loss](
                outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
            )
        return loss_map[loss](outputs, clip_targets, clip_indices, num_masks)

    def forward(self, outputs, clip_targets, frame_targets, frame_indices=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        clip_indices = self.matcher(outputs_without_aux, clip_targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in clip_targets) * outputs_without_aux["pred_logits"].shape[0]
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                clip_indices = self.matcher(aux_outputs, clip_targets)
                for loss in self.losses:
                    # fg_sim, agcl_frame, agcl_instance, calib only computed on the final layer
                    if loss in ("fg_sim", "agcl_frame", "agcl_instance", "calib"):
                        continue
                    l_dict = self.get_loss(
                        loss, aux_outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
