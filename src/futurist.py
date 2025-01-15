import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import einops
import torchvision.transforms as T
import numpy as np
from torchmetrics import JaccardIndex, Accuracy
from torchmetrics.regression import MeanSquaredError
from torchmetrics.aggregation import MeanMetric
from src.attention_masked import MaskTransformer
import math
IGNORE_LABEL = 255


class Futurist(pl.LightningModule):
    def __init__(self,args):
        super(Futurist,self).__init__()
        self.args = args
        self.sequence_length = args.sequence_length # 4
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim 
        self.heads = args.heads
        self.layers = args.layers 
        self.dropout = args.dropout
        self.patch_size = args.patch_size
        self.modality = args.modality
        self.class_colors_arr = args.class_colors_arr
        self.emb_dim = args.emb_dim
        self.shape = (self.sequence_length, (args.shape[0]//(args.downsample_factor*self.patch_size)), (args.shape[1]//(args.downsample_factor*self.patch_size)))
        self.train_mask_frames = args.train_mask_frames
        self.train_mask_mode = args.train_mask_mode
        self.masking_strategy = args.masking_strategy
        self.masking = args.masking
        self.modal_fusion = args.modal_fusion
        # Transformer - FUTURIST
        if self.modality == 'segmaps':
            self.embedding_dim = self.emb_dim[0]*(self.patch_size**2)
            self.emb = nn.Embedding(args.num_classes, self.emb_dim[0])
            self.bias = nn.Parameter(torch.zeros(args.num_classes))
        elif self.modality == 'depth':
            self.embedding_dim = self.emb_dim[0]*(self.patch_size**2)
            self.emb = nn.Embedding(args.num_classes, self.emb_dim[0])
            self.bias = nn.Parameter(torch.zeros(args.num_classes))
        elif self.modality == 'segmaps_depth':
            if self.args.modal_fusion == "concat":
                self.embedding_dim = 2*self.emb_dim[0]*(self.patch_size**2)
            elif self.args.modal_fusion == "add" and self.masking_strategy == "fully_shared":
                self.embedding_dim = self.emb_dim[0]*(self.patch_size**2)
            else:
                assert "Modal fusion not implemented"
            self.emb_s = nn.Embedding(19, self.emb_dim[0])
            self.bias_s = nn.Parameter(torch.zeros(19))
            self.emb_d = nn.Embedding(256, self.emb_dim[1])
            self.bias_d = nn.Parameter(torch.zeros(256))
        self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
                                       heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout, use_fc_bias=args.use_fc_bias, seperable_attention=args.seperable_attention,
                                       seperable_window_size=args.seperable_window_size,use_first_last=args.use_first_last, sep_head=self.args.sep_head)
        assert self.masking in ("half_half", "simple_replace", "half_half_previous")
        if self.masking in ("half_half", "half_half_previous"): 
            self.mask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.unmask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            if self.masking=="half_half":
                self.replace_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim//2)
        elif self.masking == "simple_replace":
            if self.modality in ['segmaps', 'depth']:
                self.embed = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
                self.replace_vector = nn.Parameter(torch.zeros(1, 1, 1, 1, self.hidden_dim))
                torch.nn.init.normal_(self.replace_vector, std=.02)
            elif self.modality == 'segmaps_depth':
                self.emb_input_dim = self.embedding_dim //2 if self.modal_fusion == "concat" else self.embedding_dim
                self.emb_out_dim = self.hidden_dim //2 if self.modal_fusion == "concat" else self.hidden_dim
                self.embed_segm = nn.Linear(self.emb_input_dim, self.emb_out_dim, bias=True) 
                self.embed_depth = nn.Linear(self.emb_input_dim, self.emb_out_dim, bias=True) 
                self.replace_vector_segm = nn.Parameter(torch.zeros(1, 1, 1, 1, self.emb_out_dim)) 
                torch.nn.init.normal_(self.replace_vector_segm, std=.02)
                self.replace_vector_depth = nn.Parameter(torch.zeros(1, 1, 1, 1, self.emb_out_dim))
                torch.nn.init.normal_(self.replace_vector_depth, std=.02)
                # else:
                #     assert "Modality fusion not implemented"
            else:
                assert "Modality not implemented"
            self.maskvit.fc_in = nn.Identity()
        # Necessary for evaluation
        self.valid_acc_metric = Accuracy(task="multiclass", num_classes=args.num_classes, ignore_index=IGNORE_LABEL)
        self.valid_iou_metric = JaccardIndex(task="multiclass", num_classes=args.num_classes, ignore_index=IGNORE_LABEL, average=None)
        self.mean_metric = MeanMetric()
        self.d1 = MeanMetric()
        self.d2 = MeanMetric()
        self.d3 = MeanMetric()
        self.abs_rel = MeanMetric()
        self.rmse = MeanMetric()
        self.log_10 = MeanMetric()
        self.rmse_log = MeanMetric()
        self.silog = MeanMetric()
        self.sq_rel = MeanMetric()
        self.color_per_class = nn.Parameter(torch.Tensor(args.class_colors_arr), requires_grad=False)
        self.mask_segm = None
        self.mask_depth = None
        self.save_hyperparameters()

    def get_mask_tokens(self, x, mode="arccos", mask_frames=1):
        B, sl, h, w, c = x.shape # x.shape [B,T,H,W,C]
        assert mask_frames <= sl
        if mode == "full_mask":
            if self.sequence_length == 7:
                assert mask_frames <= 3
            else:
                assert mask_frames == 1
            if self.modality in ['segmaps', 'depth']:
                mask = torch.ones(B,mask_frames,h,w, dtype=torch.bool)
            elif self.modality == 'segmaps_depth':
                mask_segm = torch.ones(B,mask_frames,h,w, dtype=torch.bool)
                mask_depth = torch.ones(B,mask_frames,h,w, dtype=torch.bool)
            else:
                assert "Modality not implemented"
        else:
            if self.args.modality in ['segmaps', 'depth']:
                r = torch.rand(B) # Batch size
            elif self.args.modality == 'segmaps_depth':
                if self.masking_strategy == "fully_indep_diff_r":
                    r = torch.rand(B,2)
                else:
                    r = torch.rand(B)
            else:
                assert "Modality not implemented"
            if mode == "linear":                # linear scheduler
                val_to_mask = r
            elif mode == "square":              # square scheduler
                val_to_mask = (r ** 2)
            elif mode == "cosine":              # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":              # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                val_to_mask = None
            # Create a mask of size [Batch,mask_frames,Height, Width] for the last frame of each sequence
            if self.modality in ['segmaps', 'depth']:
                mask = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)
            elif self.modality == 'segmaps_depth':
                if self.masking_strategy == "par_shared_excl":
                    mask_both = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)
                    mask_ids = torch.nonzero(~mask_both, as_tuple=False)
                    sh_ids = mask_ids[torch.randperm(len(mask_ids))]
                    half = sh_ids.size(0)//2
                    sel_ids1 = sh_ids[:half]
                    sel_ids2 = sh_ids[half:]
                    mask_1 = torch.zeros(B, mask_frames, h, w, dtype=torch.bool)
                    mask_2 = torch.zeros(B, mask_frames, h, w, dtype=torch.bool)
                    mask_1[sel_ids1[:,0],sel_ids1[:,1],sel_ids1[:,2],sel_ids1[:,3]] = True
                    mask_2[sel_ids2[:,0],sel_ids2[:,1],sel_ids2[:,2],sel_ids2[:,3]] = True
                    mask_segm = torch.logical_or(mask_both, mask_1)
                    mask_depth = torch.logical_or(mask_both, mask_2)
                elif self.masking_strategy == "fully_shared":
                    mask = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)
                    mask_segm = mask.clone()
                    mask_depth = mask.clone()
                elif self.masking_strategy == "fully_indep_same_r":
                    mask_segm = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)
                    mask_depth = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)
                elif self.masking_strategy == "fully_indep_diff_r":
                    mask_segm = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask[0].view(B, 1, 1, 1)
                    mask_depth = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask[1].view(B, 1, 1, 1)
            else:
                assert "Modality not implemented"
        # Create the mask for all frames, by concatenating the mask for the last frame to a tensor of zeros(no mask) for first T-1 frames
        if self.modality in ['segmaps', 'depth']:
            mask = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask], dim=1).to(x.device)
        elif self.modality == 'segmaps_depth':
            mask_segm = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask_segm], dim=1).to(x.device)
            mask_depth = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask_depth], dim=1).to(x.device)
        else:
            assert "Modality not implemented"

        if self.masking in ("half_half", "half_half_previous"):
            # Create the mask_tokens tensor
            mask_tokens = mask.unsqueeze(-1).float()*self.mask_vector.expand(B,sl,h,w,-1) + (~mask.unsqueeze(-1)).float()*self.unmask_vector.expand(B,sl,h,w,-1)
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            if self.masking == "half_half_previous":
                # Replace the embedded tokens at masked locations with the embedded tokens from the previous frames
                replace = torch.cat((torch.zeros((B,1,h,w,embedded_tokens.shape[-1]), dtype=embedded_tokens.dtype, device=embedded_tokens.device), embedded_tokens[:,:-1]), dim=1)
            else:
                # Replace the embedded tokens at masked locations with the replace vector
                replace = self.replace_vector.expand(B,sl,h,w,-1)
            embedded_tokens = torch.where(mask.unsqueeze(-1), replace, embedded_tokens)
            # Concatenate the masked tokens to the embedded tokens. Only take half of each to get the right hidden size
            final_tokens = torch.cat((embedded_tokens,mask_tokens), dim=-1)
        elif self.masking=="simple_replace":
            if self.modality in ['segmaps', 'depth']:
                # Embed the tokens
                embedded_tokens = self.embed(x) # Embedding -> Hidden
                # Replace the embedded tokens at masked locations with the replace vector
                final_tokens = torch.where(mask.unsqueeze(-1), self.replace_vector.expand(B,sl,h,w,-1), embedded_tokens)
            elif self.modality == 'segmaps_depth':
                embedded_tokens_segm = self.embed_segm(x[:,:,:,:,:self.emb_input_dim]) 
                embedded_tokens_depth = self.embed_depth(x[:,:,:,:,self.emb_input_dim:])
                final_tokens_segm = torch.where(mask_segm.unsqueeze(-1), self.replace_vector_segm.expand(B,sl,h,w,-1), embedded_tokens_segm)
                final_tokens_depth = torch.where(mask_depth.unsqueeze(-1), self.replace_vector_depth.expand(B,sl,h,w,-1), embedded_tokens_depth)
                if self.modal_fusion == "add" and self.masking_strategy == "fully_shared":
                    final_tokens = final_tokens_segm + final_tokens_depth
                if self.modal_fusion == "concat":
                    final_tokens = torch.cat([final_tokens_segm,final_tokens_depth],dim=-1)
                else:
                    assert "Modal fusion not implemented"
                self.mask_segm = mask_segm
                self.mask_depth = mask_depth
                mask = torch.logical_or(mask_segm,mask_depth)

        return final_tokens, mask

    def preprocess(self, x):
        B, T, C, H, W = x.shape   
        if self.modality == "segmaps_depth":
            x_oracle = x.clone()
            x_oracle = einops.rearrange(x, 'b t c (h k1) (w k2)  -> b t h w (c k1 k2)', k1=self.patch_size, k2=self.patch_size)
            x_s = x[:,:,0]
            x_s = self.emb_s(x_s.to(torch.int))
            x_s = einops.rearrange(x_s, 'b t (h k1) (w k2) c -> b t h w (c k1 k2)', k1=self.patch_size, k2=self.patch_size)
            x_dp = x[:,:,1]
            x_dp = self.emb_d(x_dp.to(torch.int))
            x_dp = einops.rearrange(x_dp, 'b t (h k1) (w k2) c -> b t h w (c k1 k2)', k1=self.patch_size, k2=self.patch_size)
            x = torch.cat([x_s,x_dp],dim=-1)
        else:
            x_oracle = x.clone()
            x_oracle = einops.rearrange(x, 'b t c (h k1) (w k2)  -> b t h w (c k1 k2)', k1=self.patch_size, k2=self.patch_size)
            x = self.emb(x.to(torch.int))
            x = x.squeeze(2)
            x = einops.rearrange(x, 'b t (h k1) (w k2) c -> b t h w (c k1 k2)', k1=self.patch_size, k2=self.patch_size)
        return x, x_oracle

    def decode(self, x):
        if self.modality == "segmaps_depth":
            x = einops.rearrange(x , 'b t h w (c k1 k2) -> b t (h k1) (w k2) c', k1=self.patch_size, k2=self.patch_size)
            if self.modal_fusion == "concat":
                x_s = x[:,:,:,:,:self.emb_dim[0]]
                x_d = x[:,:,:,:,self.emb_dim[0]:]
                x_s = torch.matmul(x_s, self.emb_s.weight.T) + self.bias_s
                x_d = torch.matmul(x_d, self.emb_d.weight.T) + self.bias_d
            elif self.modal_fusion == "add":
                x_s = torch.matmul(x, self.emb_s.weight.T) + self.bias_s
                x_d = torch.matmul(x, self.emb_d.weight.T) + self.bias_d
            x_s = torch.movedim(x_s,-1,2)
            x_d = torch.movedim(x_d,-1,2)
            x = torch.cat([x_s,x_d],dim=2)
        else:
            x = einops.rearrange(x , 'b t (c k1 k2) h w -> b t (h k1) (w k2) c', k1=self.patch_size, k2=self.patch_size)
            x = torch.matmul(x, self.emb.weight.T) + self.bias
            x = torch.movedim(x,-1,2)
        return x
    
    def forward_loss(self, x_pred, mask, x_oracle):
        B = x_pred.size(0)
        if self.modality in ['segmaps', 'depth']:
            x_pred = x_pred[mask]
            x_oracle = x_oracle[mask]
        else:
            x_pred_s = x_pred[self.mask_segm]
            x_pred_d = x_pred[self.mask_depth]
            x_oracle_s = x_oracle[self.mask_segm]
            x_oracle_d = x_oracle[self.mask_depth]
        if self.modality in ['segmaps', 'depth']:
            ig_index = 255 if self.args.modality == 'segmaps' else 0
            x_oracle = einops.rearrange(x_oracle, 'b (c p1 p2)-> b c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
            x_pred = einops.rearrange(x_pred, 'b (c p1 p2)  -> b (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
            x_out = torch.matmul(x_pred, self.emb.weight.T) + self.bias
            CE = nn.CrossEntropyLoss(ignore_index=ig_index)
            loss = CE(torch.movedim(x_out,-1,1), x_oracle.squeeze(1).to(torch.long))
        elif self.modality == "segmaps_depth":
            x_oracle_s = einops.rearrange(x_oracle_s, 'b (c p1 p2) -> b c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
            x_oracle_d = einops.rearrange(x_oracle_d, 'b (c p1 p2) -> b c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
            x_pred_s = einops.rearrange(x_pred_s, 'b (c p1 p2) -> b (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
            x_pred_d = einops.rearrange(x_pred_d, 'b (c p1 p2) -> b (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
            x_oracle_s = x_oracle_s[:,0]
            x_oracle_d = x_oracle_d[:,1]
            if self.modal_fusion == "concat":
                x_s = x_pred_s[:,:,:self.emb_dim[0]]
                x_d = x_pred_d[:,:,self.emb_dim[0]:]
                x_s = torch.matmul(x_s, self.emb_s.weight.T) + self.bias_s
                x_d = torch.matmul(x_d, self.emb_d.weight.T) + self.bias_d
            else: # Add
                x_s = torch.matmul(x_pred_s, self.emb_s.weight.T) + self.bias_s
                x_d = torch.matmul(x_pred_d, self.emb_d.weight.T) + self.bias_d
            CEs = nn.CrossEntropyLoss(ignore_index=255)
            CEd = nn.CrossEntropyLoss(ignore_index=0)
            loss_s = self.args.w_s*CEs(torch.movedim(x_s,-1,1), x_oracle_s.squeeze(1).to(torch.long))
            loss_d = (1-self.args.w_s)*CEd(torch.movedim(x_d,-1,1), x_oracle_d.squeeze(1).to(torch.long))
            self.log("loss_segm", loss_s, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
            self.log("loss_depth", loss_d, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
            loss = loss_s + loss_d
        return loss


    def sample(self, x, mask_frames=1):
        self.maskvit.eval()
        with torch.no_grad():
            x, x_oracle = self.preprocess(x)
            masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
            mask = mask.to(x.device)
            if self.args.vis_attn:
                _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask, x_oracle)
            else:
                loss, final_tokens,  = self.forward(x, masked_soft_tokens, mask, x_oracle)
            prediction = self.decode(final_tokens)
            return prediction, loss
    
        
    def forward(self, x, masked_x, mask=None, x_oracle=None):
        if self.args.vis_attn:
            x_pred,  attn = self.maskvit(masked_x,return_attn=True)
        else:
            x_pred = self.maskvit(masked_x)
        x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c',sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        loss = self.forward_loss(x_pred=x_pred, mask=mask, x_oracle=x_oracle)
        if self.args.vis_attn:
            return loss, x_pred, attn
        else:
            return loss, x_pred

    def training_step(self, x, batch_idx):
        if isinstance(x, (list, tuple)):
            x = x[0]
        B = x.shape[0]
        x, x_oracle = self.preprocess(x)
        # Mask the encoded tokens
        if self.sequence_length == 7:
            train_mask_frames = torch.randint(1, 4, (1,)) if self.training else 3
        else:
            train_mask_frames = self.train_mask_frames if self.training else 1
        masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=train_mask_frames) # masked_x.shape [B,T,H,W,C], mask.shape [B,T,H,W,1]
        if self.args.vis_attn:
            loss, _, _ = self.forward(x, masked_x, mask, x_oracle)
        else:
            loss, _ = self.forward(x, masked_x, mask, x_oracle)
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("Train/lr", lr, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        mask_ratio = mask[:,-1].float().mean().item() * 100
        self.log("Train/mask_ratio", mask_ratio, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.eval_mode:
            return self.evaluation_step(batch, batch_idx)
        B = batch[0].shape[0]
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, batch_size=B, logger=True, on_step=True)

    def update_metrics(self, pred, gt, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
        valid_pixels = gt > 0
        pred = pred[valid_pixels]
        gt = gt[valid_pixels]
        pred = torch.clamp(pred, min=1)
        pred = pred.float()/255.0
        gt = gt.float()/255.0
        thresh = torch.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).float().mean()
        d2 = (thresh < 1.25 ** 2).float().mean()
        d3 = (thresh < 1.25 ** 3).float().mean()
        d1_m.update(d1)
        d2_m.update(d2)
        d3_m.update(d3)
        abs_rel = torch.mean(torch.abs(gt - pred) / gt)
        sq_rel = torch.mean(((gt - pred) ** 2) / gt)
        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.float().mean())
        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())
        err = torch.log(pred) - torch.log(gt)
        silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
        log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
        abs_rel_m.update(abs_rel)
        rmse_m.update(rmse)
        log_10_m.update(log_10)
        rmse_log_m.update(rmse_log)
        silog_m.update(silog)
        sq_rel_m.update(sq_rel)

    def compute_metrics(self, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
        d1 = d1_m.compute()
        d2 = d2_m.compute()
        d3 = d3_m.compute()
        abs_rel = abs_rel_m.compute()
        rmse = rmse_m.compute()
        log_10 = log_10_m.compute()
        rmse_log = rmse_log_m.compute()
        silog = silog_m.compute()
        sq_rel = sq_rel_m.compute()
        return d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel
        
    def reset_metrics(self, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
        d1_m.reset()
        d2_m.reset()
        d3_m.reset()
        abs_rel_m.reset()
        rmse_m.reset()
        log_10_m.reset()
        rmse_log_m.reset()
        silog_m.reset()
        sq_rel_m.reset()

    def evaluation_step(self, batch, batch_idx):
        if self.modality in ['segmaps', 'depth']:
            data_tensor, gt = batch
        elif self.modality == 'segmaps_depth':
            data_tensor, gt_seg, gt_depth = batch
        if not self.args.evaluate_baseline:
            if self.args.eval_midterm:
                if self.sequence_length < 7:
                    for i in range(3):
                        with torch.no_grad():
                            if i<2:
                                samples, loss = self.sample(data_tensor)
                                if self.modality in ['segmaps', 'depth']:
                                    samples = samples.argmax(dim=2).unsqueeze(2)
                                    data_tensor[:,-1] = samples[:,-1]
                                    data_tensor[:,0:-1] = data_tensor[:,1:].clone()
                                elif self.modality == 'segmaps_depth':
                                    samples_segm = samples[:,-1,:19].argmax(dim=1).unsqueeze(1)
                                    samples_depth = samples[:,-1,19:].argmax(dim=1).unsqueeze(1)
                                    data_tensor[:,-1] = torch.cat([samples_segm, samples_depth], dim=1)
                                    data_tensor[:,0:-1] = data_tensor[:,1:].clone()
                            else:
                                samples, loss = self.sample(data_tensor)                
                else:
                    for i in range(3):
                        with torch.no_grad():
                            if i<2:
                                samples, loss = self.sample(data_tensor,mask_frames=3-i)
                                if self.modality in ['segmaps', 'depth']:
                                    samples = samples.argmax(dim=2).unsqueeze(2)
                                    data_tensor[:,-(3-i)] = samples[:,-(3-i)]
                                elif self.modality == 'segmaps_depth':
                                    samples_segm = samples[:,-(3-i),:19].argmax(dim=1).unsqueeze(1)
                                    samples_depth = samples[:,-(3-i),19:].argmax(dim=1).unsqueeze(1)
                                    data_tensor[:,-(3-i)] = torch.cat([samples_segm, samples_depth], dim=1)
                            else:
                                samples, loss = self.sample(data_tensor,mask_frames=1)
            else:
                # Short Term Evaluation
                with torch.no_grad():
                    x, x_oracle = self.preprocess(data_tensor)
                    samples, loss = self.sample(data_tensor)
        else:
            samples = data_tensor
            samples[:,-1] = samples[:,-2]
            loss = 0 # Just to avoid error for baseline evaluation
        self.mean_metric.update(loss)
        mean_loss = self.mean_metric.compute()
        if 'segmaps' in self.args.modality:
            pred_seg = samples[:,-1]
            if self.args.modality == 'segmaps_depth' and not self.args.evaluate_baseline:
                pred_seg = pred_seg[:,:19]
            if not self.args.evaluate_baseline:
                pred_seg = pred_seg.argmax(dim=1)
            else:
                pred_seg = pred_seg[:,0]
            upsample = T.Resize(self.args.shape, T.InterpolationMode.NEAREST)
            pred_seg = upsample(pred_seg)
            gt_seg = gt_seg.long()
            self.valid_iou_metric.update(pred_seg, gt_seg)
            self.valid_acc_metric.update(pred_seg, gt_seg)    
            IoU = self.valid_iou_metric.compute()
            Acc = self.valid_acc_metric.compute()
            mIoU = torch.mean(IoU)
            MO_mIoU = torch.mean(IoU[11:])
            self.log('val/mIoU', mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/MO_mIoU', MO_mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/Acc', Acc, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            print(f'Iter {batch_idx}: mIoU={mIoU} - MO_mIoU={MO_mIoU}')
        if 'depth' in self.args.modality:
            pred_depth = samples[:,-1]
            if self.args.modality == 'segmaps_depth' and not self.args.evaluate_baseline:
                pred_depth = pred_depth[:,19:]
            if not self.args.evaluate_baseline:
                pred_depth = pred_depth.argmax(dim=1)
            else:
                if self.args.modality == 'segmaps_depth':
                    pred_depth = pred_depth[:,1]
                else:
                    pred_depth = pred_depth[:,0]
            upsample = T.Resize(self.args.shape, T.InterpolationMode.NEAREST)
            pred_depth = upsample(pred_depth)
            self.update_metrics(pred_depth, gt_depth, self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = self.compute_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            self.log('val/d1', d1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/d2', d2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/d3', d3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/abs_rel', abs_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/log_10', log_10, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse_log', rmse_log, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/silog', silog, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/sq_rel', sq_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        self.log('val/loss', loss, prog_bar=True, batch_size=data_tensor.shape[0], sync_dist=True, on_step=False, on_epoch=True, logger=True)
        
        
        

    def on_validation_epoch_end(self):
        if self.args.eval_mode:
            IoU = self.valid_iou_metric.compute()
            Acc = self.valid_acc_metric.compute()
            mean_loss = self.mean_metric.compute()
            print("Mean_Loss = %10f" % mean_loss)
            mean_loss = self.mean_metric.reset()
            if 'segmaps' in self.args.modality:
                mIoU = torch.mean(IoU)
                MO_mIoU = torch.mean(IoU[11:])
                print("mIoU = %10f" % (mIoU*100))
                print("MO_mIoU = %10f" % (MO_mIoU*100))
                print("Acc = %10f" % (Acc * 100))
                print("class_IoU = ", IoU)
                self.log_dict({"mIoU": mIoU * 100, "MO_mIoU": MO_mIoU * 100, "Acc": Acc * 100}, logger=True, prog_bar=True)
                self.log_dict({"road": IoU[0], "sidewalk": IoU[1], "building": IoU[2], "wall": IoU[3], "fence": IoU[4], "pole": IoU[5], "traffic light": IoU[6], "traffic sign": IoU[7], "vegetation": IoU[8], "terrain": IoU[9], "sky": IoU[10], "person": IoU[11], "rider": IoU[12], "car": IoU[13], "truck": IoU[14], "bus": IoU[15], "train": IoU[16], "motorcycle": IoU[17], "bicycle": IoU[18]}, logger=True, prog_bar=True)
                self.valid_iou_metric.reset()
                self.valid_acc_metric.reset()                
            if 'depth' in self.args.modality:
                d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = self.compute_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
                print("d1=%10f" % (d1), "d2=%10f" % (d2), "d3=%10f" % (d3), "abs_rel=%10f" % (abs_rel), "rmse=%10f" % (rmse), "log_10=%10f" % (log_10), "rmse_log=%10f" % (rmse_log), "silog=%10f" % (silog), "sq_rel=%10f" % (sq_rel))
                self.log_dict({"d1":d1, "d2":d2, "d3":d3, "abs_rel":abs_rel, "rmse":rmse, "log_10":log_10, "rmse_log":rmse_log, "silog":silog, "sq_rel":sq_rel}, logger=True, prog_bar=True)
                self.reset_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]