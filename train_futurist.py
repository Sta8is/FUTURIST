from src.data import CS_VideoData 
from src.futurist import Futurist
import pytorch_lightning as pl
import torch 
import argparse
import os
import yaml
from pytorch_lightning.strategies import DDPStrategy
import numpy as np


def get_cityscapes_class_colors_arr():
    with open('configs/cityscapes_colors.yml', 'r') as file:
        # Parse the YAML file into a Python dictionary
        color_data = yaml.safe_load(file)
    # Now 'data' contains the contents of your YAML file as a dictionary
    obj_colors = {d["id"]: d["color"] for d in color_data}
    # print(obj_colors.items())
    class_colors_arr = np.array([obj_colors[k] for k in sorted(obj_colors.keys())],dtype=np.uint8)
    return class_colors_arr

def parse_list(x):
    return list(map(int, x.split(',')))

parser = argparse.ArgumentParser()
# Data Parameters
parser.add_argument('--data_path', type=str, default="/storage/e.karypidis/cityscapes/leftImg8bit_sequence_segmaps")
parser.add_argument('--dst_path', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_workers_val', type=int, default=4, 
        help="(Optional) number of workers for the validation set dataloader. If None (default) it is the same as num_workers.")
parser.add_argument('--sequence_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--shape', type=tuple, default=(1024,2048))
parser.add_argument('--downsample_factor', type=int, default=4) 
parser.add_argument('--random_crop', action='store_true', default=False)
parser.add_argument('--random_horizontal_flip', action='store_true', default=False)
parser.add_argument('--random_time_flip', action='store_true', default=False)
parser.add_argument('--timestep_augm', type=list, default=None, help="Probabilities for each timestep to be selected for augmentation starting from timestep 2 to \length of prob list e.g. [0.1,0.6,0.1,0.1,0.1] for timesteps [2,3,4,5,6]. If None, timestep [2,3,4] are selected with equal probability")
parser.add_argument("--no_timestep_augm", action='store_true', help="If True, no timestep augmentation is used (i.e., the num_frames_skip is always equal to 2 during training.)")
parser.add_argument('--modality', type=str, default='segmaps_depth', choices=['segmaps', 'depth', 'segmaps_depth'])
parser.add_argument("--use_fc_bias", action='store_true', help="Use bias for the fc_in and fc_out layers.")
parser.add_argument("--num_classes", type=int, default=256)
# GIVT Masked parmeters
parser.add_argument('--hidden_dim', type=int, default=1536)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--layers', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--loss_all_tokens', action='store_true', default=False)
parser.add_argument('--attn_dropout', type=float, default=0.3)
parser.add_argument("--accum_iter", type=int, default=1)
parser.add_argument('--masking', type=str, default='simple_replace', choices=["half_half", "simple_replace", "half_half_previous"])
parser.add_argument('--train_mask_mode', type=str, default='arccos', choices=["full_mask", "arccos", "linear", "cosine", "square"])
parser.add_argument('--modal_fusion', type=str, default='concat', choices=['concat', 'add'])
parser.add_argument('--masking_strategy', type=str, default='par_shared_excl', choices=["fully_shared", "par_shared_excl", "fully_indep_same_r","fully_indep_diff_r"])
parser.add_argument("--seperable_attention", action='store_true', default=False)
parser.add_argument("--seperable_window_size", type=int, default=1)
parser.add_argument("--train_mask_frames", type=int, default=1)
parser.add_argument("--use_first_last", action='store_true', default=False)
parser.add_argument("--emb_dim", type=parse_list, default=[10], help="Embedding dimensions for the input tokens. If multiple values are provided,first number is for segmentation and the second for depth")
parser.add_argument('--w_s', type=float, default=0.5, help="Weight for the segmentation loss. Depth loss weight is 1-w_s")
parser.add_argument('--sep_head', action='store_true', default=False)
# training parameters
parser.add_argument("--max_epochs", type=int, default=800) 
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--precision", type=str, default="32-true",choices=["16-true","16-mixed","32-true", "32"])
parser.add_argument("--ckpt", type=str, default=None, help="Path of a checkpoint to resume training")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--lr_base", type=float, default=4e-5)
parser.add_argument("--gclip", type=float, default=1.0)
parser.add_argument("--evaluate", action='store_true', default=False)
parser.add_argument("--vis_attn", action='store_true', default=False)
parser.add_argument("--eval_last", action='store_true', default=False)
parser.add_argument("--eval_ckpt_only", action='store_true', default=False)
parser.add_argument('--eval_mode_during_training', action='store_true', help="if activated (True) it uses the evaluation mode (i.e., step-by-step prediction and mIoU computation) during the training loop")
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--use_val_to_train", action='store_true', default=False)
parser.add_argument("--use_train_to_val", action='store_true', default=False)
parser.add_argument("--evaluate_baseline", action='store_true', default=False)
parser.add_argument("--eval_midterm", action='store_true', default=False)

args = parser.parse_args()


args.eval_mode = args.eval_mode_during_training
pl.seed_everything(args.seed, workers=True)

data = CS_VideoData(arguments=args,subset="train",batch_size=args.batch_size)


args.class_colors_arr = get_cityscapes_class_colors_arr()
if args.precision == "32":
    args.precision = 32

if args.num_gpus > 1:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        args.node = args.rank // gpus_per_node
    else:
        args.rank = 0
        args.world_size = args.num_gpus
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device(args.gpu)
else:
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.node = 0
    args.device = torch.device("cuda:"+str(args.gpu))

print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
args.max_steps = (args.max_epochs * (len(data.train_dataloader()) // (args.num_gpus * args.accum_iter)))
args.effective_batch_size = args.batch_size * args.world_size * args.accum_iter
args.lr = (args.lr_base * args.effective_batch_size) / 8 # args.lr_base is specified for an effective batch-size of 8
print(f'Effective batch size:{args.effective_batch_size} lr_base={args.lr_base} lr={args.lr} max_epochs={args.max_epochs} - max_steps={args.max_steps}')

futurist = Futurist(args)

callbacks = []
monitor_metric = 'val/mIoU' if 'segmaps' in args.modality else 'val/abs_rel'
monitor_mode = 'max' if 'segmaps' in args.modality else 'min'
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor_metric, mode=monitor_mode, save_top_k=1, save_last=True)
callbacks.append(checkpoint_callback)
if args.dst_path is None:
    args.dst_path = os.getcwd()
if args.max_epochs < args.eval_freq:
    args.eval_freq = 1
trainer = pl.Trainer(
    accelerator="gpu",
    strategy=(DDPStrategy(find_unused_parameters=False) if args.num_gpus > 1 else 'auto'),
    devices=args.num_gpus,
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    gradient_clip_val=args.gclip,
    default_root_dir=args.dst_path,
    precision=args.precision,
    log_every_n_steps=5,
    check_val_every_n_epoch=args.eval_freq,
    accumulate_grad_batches=args.accum_iter)

if not args.eval_ckpt_only:
    trainer.fit(futurist,data)
else:
    args.evaluate = True
    args.eval_last = True
    checkpoint_callback.last_model_path = args.ckpt
    data = CS_VideoData(arguments=args,subset="train",batch_size=args.batch_size)

# Evaluation
if args.evaluate:
    args.eval_mode = True
    if not args.eval_last:
        print("Loading best model")
        checkpoint_path = checkpoint_callback.best_model_path
    else:
        print("Loading last model")
        checkpoint_path = checkpoint_callback.last_model_path

    print(f'checkpoint_path = {checkpoint_path}')
    futurist = Futurist.load_from_checkpoint(checkpoint_path,args=args,strict=False)
    print("-----------MaskedGIVT.eval_mode = ",futurist.args.eval_mode)
    futurist.to(args.device)
    futurist.eval()

    val_data_loader = data.val_dataloader()
    out_metrics = trainer.validate(model=futurist, dataloaders=val_data_loader)
    if 'segmaps' in args.modality:
        mIoU = out_metrics[0]["mIoU"]
        MO_mIoU = out_metrics[0]["MO_mIoU"]
        road_IoU = out_metrics[0]["road"]
        sidewalk_IoU = out_metrics[0]["sidewalk"]
        building_IoU = out_metrics[0]["building"]
        wall_IoU = out_metrics[0]["wall"]
        fence_IoU = out_metrics[0]["fence"]
        pole_IoU = out_metrics[0]["pole"]
        traffic_light_IoU = out_metrics[0]["traffic light"]
        traffic_sign_IoU = out_metrics[0]["traffic sign"]
        vegetation_IoU = out_metrics[0]["vegetation"]
        terrain_IoU = out_metrics[0]["terrain"]
        sky_IoU = out_metrics[0]["sky"]
        person_IoU = out_metrics[0]["person"]
        rider_IoU = out_metrics[0]["rider"]
        car_IoU = out_metrics[0]["car"]
        truck_IoU = out_metrics[0]["truck"]
        bus_IoU = out_metrics[0]["bus"]
        train_IoU = out_metrics[0]["train"]
        motorcycle_IoU = out_metrics[0]["motorcycle"]
        bicycle_IoU = out_metrics[0]["bicycle"]
        class_IoU = [road_IoU, sidewalk_IoU, building_IoU, wall_IoU, fence_IoU, pole_IoU, traffic_light_IoU, traffic_sign_IoU, vegetation_IoU, terrain_IoU, sky_IoU, person_IoU, rider_IoU, car_IoU, truck_IoU, bus_IoU, train_IoU, motorcycle_IoU, bicycle_IoU]
        #if args.rank == 0:
        #    mIoU, MO_mIoU = calculate_iou(futurist, val_data_loader, args.class_colors_arr, args)
        if args.rank == 0:
            # Save mIoU and MO_mIoU to a text file
            result_path = os.path.join(trainer.log_dir, 'results_seg.txt')
            with open(result_path, 'w') as f:
                f.write(f'mIoU: {mIoU}\n')
                f.write(f'MO_mIoU: {MO_mIoU}\n')
                f.write('road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle\n')
                f.write(f'{road_IoU}, {sidewalk_IoU}, {building_IoU}, {wall_IoU}, {fence_IoU}, {pole_IoU}, {traffic_light_IoU}, {traffic_sign_IoU}, {vegetation_IoU}, {terrain_IoU}, {sky_IoU}, {person_IoU}, {rider_IoU}, {car_IoU}, {truck_IoU}, {bus_IoU}, {train_IoU}, {motorcycle_IoU}, {bicycle_IoU}\n')
            print(f'Results saved at: {result_path}')
    if 'depth' in args.modality:
        d1 = out_metrics[0]["d1"]
        d2 = out_metrics[0]["d2"]
        d3 = out_metrics[0]["d3"]
        abs_rel = out_metrics[0]["abs_rel"]
        rmse = out_metrics[0]["rmse"]
        rmse_log = out_metrics[0]["rmse_log"]
        sq_rel = out_metrics[0]["sq_rel"]
        log_10 = out_metrics[0]["log_10"]
        silog = out_metrics[0]["silog"]
        if args.rank == 0:
            # Save metrics to a text file
            result_path = os.path.join(trainer.log_dir, 'results_dp.txt')
            with open(result_path, 'w') as f:
                f.write(f'd1: {d1}\n')
                f.write(f'd2: {d2}\n')
                f.write(f'd3: {d3}\n')
                f.write(f'abs_rel: {abs_rel}\n')
                f.write(f'rmse: {rmse}\n')
                f.write(f'rmse_log: {rmse_log}\n')
                f.write(f'sq_rel: {sq_rel}\n')
                f.write(f'log_10: {log_10}\n')
                f.write(f'silog: {silog}\n')
            print(f'Results saved at: {result_path}')
