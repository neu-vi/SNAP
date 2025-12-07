import torch
import argparse
import os
from datetime import datetime
from datasets import build_dataset_single_mask
from datasets import collate_utils
from dataloaders.multidataset_loader import MultiDataset
from src.snap import SNAP
from utils.dist_utils import get_dist_info, create_logger
from utils.torch_helpers import all_to_device
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import torch.nn.functional as F
from conf_input import semantic_kitti, nuscenes, pandaset, scannet, s3dis, scannetpp, kitti360, stpls3d, dales, kitti360full, s3disfull, replica, hm3d, matterport, urbanbis, kitti360_ss, waymo
import torch.distributed as dist
import pdb
import copy
import numpy as np
from torch.cuda.amp import autocast
import pyvista as pv

# Import clip to generate text embeddings for each class in a dataset
import clip

def get_args_parser():
    parser = argparse.ArgumentParser()

    # Tensorboard summary name
    # parser.add_argument('--name', default="Tensorboard_logs/Modified_NeuFlow gfdee_2_loss", type=str)
    parser.add_argument('--exp_name', default="test_single_mask", type=str)
    parser.add_argument('--logdir', default="logs/", type=str)
    parser.add_argument('--TIMEIT', default=False, type=bool)
    parser.add_argument('--dev', action='store_true', default=False)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--enable_amp', action='store_true', default=False)
    parser.add_argument('--val_only', default=False, action='store_true')

    
    # Model
    parser.add_argument('--log_mem', action='store_true', default=False)
    parser.add_argument('--num_prompt_points', default=32, type=int)
    parser.add_argument('--num_object_points', default=10, type=int)
    parser.add_argument('--num_merge', default=1, type=int)
    parser.add_argument('--overfit', action='store_true', default=False)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--use_pdnorm', action='store_true', default=False)
    parser.add_argument('--use_random_clicks', action='store_true', default=False)
    parser.add_argument('--mask_threshold', default=0.5, type=float)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--use_localized_loss', action='store_true', default=False)
    parser.add_argument('--iterative', default=False, action='store_true')

    # dataset
    parser.add_argument('--checkpoint_dir', default="checkpoints/", type=str)
    parser.add_argument('--stage', default=['kitti'], type=str, nargs='+')
    parser.add_argument('--val_dataset', default=[''], type=str, nargs='+')

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--epochs', default=500, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)

    # distributed training
    # parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true', default=False)

    # Output
    parser.add_argument('--save_output', default=False, type=bool)

    return parser

def reduce_tensor(tensor):
    # Reduces tensor across all GPUs
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def reduce_tensor_sum(tensor):
    # Reduces tensor across all GPUs
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def load_model_weights(model, checkpoint_path):
    # Load trained weights to the model
    checkpoint = torch.load(checkpoint_path)

    state_dict = {}
    for k, v in checkpoint['model'].items():
        state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    return model

def random_sample_clicks(preds, masks, coords, offset, threshold):
    batch_offset = [0] + list(offset)
    batch_size = len(masks)
    new_clicks = []
    for i in range(batch_size):
        pred = preds[i].sigmoid() > threshold
        mask = masks[i]
        coords_i = coords[batch_offset[i]:batch_offset[i+1]]
        num_obj = len(mask)
        for idx in range(num_obj):
            pred_idx = pred[idx]
            mask_idx = mask[idx]
            gt_indices = torch.where(mask_idx == 1)[0]
            pred_ = pred_idx[gt_indices]
            error_indices = torch.where(pred_ == 0)[0]
            if len(error_indices) == 0:
                select_idx = np.random.choice(gt_indices.cpu(), 1, replace=False)[0]
            else:
                error_idx = np.random.choice(error_indices.cpu(), 1, replace=False)[0]
                select_idx = gt_indices[error_idx]
            assert mask_idx[select_idx] == 1
            new_clicks.append(coords_i[select_idx].unsqueeze(0))
    new_clicks = torch.stack(new_clicks)
    return new_clicks

def plot_pointcloud_all(args, data_dict, seg_logits_prompt_list_dict, prompt_point_list_dict, iou_dict, idx, threshold=0.5):
    """
    data_dict: dictionary containing the input data
    seg_logits_prompt_list_dict: dict of list segmentation logits for each click iteration
    prompt_point_list_dict: dict of list prompt points for each click iteration
    """

    points = data_dict['coord'].cpu().numpy()
    pointcloud = pv.PolyData(points)
    pc1 = pointcloud.copy()
    pc2 = pointcloud.copy()
    pc3 = pointcloud.copy()
    pc4 = pointcloud.copy()

    if 'color' in data_dict:
        colors = data_dict['color'].cpu().numpy()
        colors = (colors+1)/2
        colors = colors * 0.7 + [0.1, 0.1, 0.1] # Grey out the colors
    else:
        colors = np.zeros_like(points) + [0.7, 0.7, 0.7]

    # Create a PyVista plotter
    plotter = pv.Plotter(shape=(2, 2))

    if args.stage[0] in ["kitti", "nuScenes", "pandaset"]:
        prompt_point_radius = 0.3
    elif args.stage[0] == "partnet":
        prompt_point_radius = 0.05
    elif args.stage[0] in ["scannet", "scanrefer", "scannetpp", "scannet-block"]:
        prompt_point_radius = 0.07
    elif args.stage[0] in ["stpls3d"]:
        prompt_point_radius = 0.1

    ## Plot the ground-truth mask
    plotter.subplot(0, 0)
    gt_mask = data_dict['masks'][0].cpu().numpy() # Shape -> M, N
    pc1_colors = colors.copy()
    
    # Iterate over all the masks
    mask_color_list = []
    prompt_point_color_list = []
    for i in range(gt_mask.shape[0]):
        # Create a random color for each mask
        color = np.random.rand(3)
        mask_color = color * 0.5 + 0.5 # Lighten the color
        mask_color_list.append(mask_color)
        prompt_point_color_list.append(color)

        pc1_colors[gt_mask[i, :] == 1] = mask_color

    pc1['RGB'] = pc1_colors
    plotter.add_text("Ground-Truth Mask", font_size=10, color='black', position='upper_edge')
    plotter.add_mesh(
        pc1, 
        scalars='RGB',
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )

    ## Plot the click masks
    for click_key in [1, 5, 10]:
        if click_key == 1:
            plotter.subplot(0, 1)
            pc_interest = pc2
        elif click_key == 5:
            plotter.subplot(1, 0)
            pc_interest = pc3
        elif click_key == 10:
            plotter.subplot(1, 1)
            pc_interest = pc4
        
        pc_colors = colors.copy()

        # Get all the predicted masks and prompt points for the current sample_iter
        seg_logits_prompt_list = seg_logits_prompt_list_dict[click_key]
        prompt_point_list = prompt_point_list_dict[click_key]
        iou_list = iou_dict[click_key]
        # pdb.set_trace()
        # Iterate over all the predicted masks and prompt points
        for i in range(len(seg_logits_prompt_list)):
            pred_mask_i = (seg_logits_prompt_list[i][0].sigmoid().cpu().numpy() > threshold).flatten()
            pc_colors[pred_mask_i==1] = mask_color_list[i]

            prompt_points_i = prompt_point_list[i].cpu().numpy()
            for j in range(prompt_points_i.shape[1]):
                plotter.add_mesh(
                    pv.Sphere(center=prompt_points_i[0, j, :], radius=prompt_point_radius),
                    color=prompt_point_color_list[i]
                )

        pc_interest['RGB'] = pc_colors

        # Compute mean IoU for the current click iteration
        mean_iou = torch.stack(iou_list).mean().cpu().numpy().item()
        # print(f"{click_key}-Click Mask, Mean IoU: {mean_iou:.2f}")
        plotter.add_text(f"{click_key}-Click Mask, Mean IoU: {mean_iou:.2f}", font_size=10, color='black', position='upper_edge')

        plotter.add_mesh(
            pc_interest,
            scalars='RGB',
            rgb=True,
            point_size=5,
            render_points_as_spheres=True
        )

    # Link the views to synchronize camera movements
    plotter.link_views()

    # Display the plot
    plotter.show()

def plot_pointcloud(args, data_dict, seg_logits_prompt_list, prompt_point_list, iou_list, idx, threshold=0.5):
    """
    data_dict: dictionary containing the input data
    seg_logits_prompt_list: list of segmentation logits for each click iteration
    prompt_point_list: list of prompt points for each click iteration
    """
    color_list = {
        0: (149,186,180),
        1: (107, 89, 99),

    }

    darkened_color_list = {
        0: (197, 212, 201),
        1: (145, 128, 133),

    }
    # pdb.set_trace()
    points = data_dict['coord'].cpu().numpy()
    pointcloud = pv.PolyData(points)
    pc1 = pointcloud.copy()
    pc2 = pointcloud.copy()
    pc3 = pointcloud.copy()
    pc4 = pointcloud.copy()

    if 'color' in data_dict:
        colors = data_dict['color'].cpu().numpy()
        colors = (colors+1)/2
        colors = colors * 0.7 + [0.1, 0.1, 0.1] # Grey out the colors
    else:
        colors = np.zeros_like(points) + [0.7, 0.7, 0.7]

    num_clicks = len(seg_logits_prompt_list)

    # Create a PyVista plotter
    # plotter = pv.Plotter(shape=(2, 2), off_screen=True)
    plotter = pv.Plotter(shape=(2, 2))
    prompt_point_radius = 0.1 if args.stage[0] == "kitti" else 0.1

    prompt_point_radius = 0.05 if args.stage[0]=="partnet" else 0.1

    # Choose a random index from the color list
    # random_idx = np.random.choice([0, 1])
    
    # # pdb.set_trace()
    # mask_color = darkened_color_list[random_idx]
    # prompt_point_color = color_list[random_idx]

    # Put the color as bright light green
    prompt_point_color = (0.0, 0.7815, 0.0)  # "light green" - Light Green
    # Put the color as bright dark green
    mask_color = (0.0, 0.5216, 0.0)  # "dark green" - Dark Green

    ## Plot the ground-truth mask 
    plotter.subplot(0, 0)
    gt_mask = data_dict['masks'][0].cpu().numpy().flatten()
    pc1_colors = colors.copy()
    pc1_colors[gt_mask == 1] = mask_color
    pc1['RGB'] = pc1_colors
    plotter.add_text("Ground-Truth Mask", font_size=10, color='black', position='upper_edge')
    plotter.add_mesh(
        pc1, 
        scalars='RGB',
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )

    ## Plot the 1-click masks
    plotter.subplot(0, 1)
    pred_mask_1 = (seg_logits_prompt_list[0][0].sigmoid().cpu().numpy() > threshold).flatten()
    pc2_colors = colors.copy()
    pc2_colors[pred_mask_1==1] = mask_color
    pc2['RGB'] = pc2_colors
    plotter.add_text(f"1-Click Mask, IoU: {iou_list[0].cpu().numpy().item():.2f}", font_size=10, color='black', position='upper_edge')
    plotter.add_mesh(
        pc2,
        scalars='RGB',
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )

    # Also add the prompt points
    prompt_points = prompt_point_list[0].cpu().numpy()
    for i in range(prompt_points.shape[1]):
        plotter.add_mesh(
            pv.Sphere(center=prompt_points[0, i, :], radius=prompt_point_radius),
            color=prompt_point_color
        )
    
    ## Plot the 5-click masks
    plotter.subplot(1, 0)
    pred_mask_5 = (seg_logits_prompt_list[4][0].sigmoid().cpu().numpy() > threshold).flatten()
    pc3_colors = colors.copy()
    pc3_colors[pred_mask_5==1] = mask_color
    pc3['RGB'] = pc3_colors
    plotter.add_text(f"5-Click Mask, IoU: {iou_list[4].cpu().numpy().item():.2f}", font_size=10, color='black', position='upper_edge')
    plotter.add_mesh(
        pc3,
        scalars='RGB',
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )

    # Also add the prompt points
    prompt_points = prompt_point_list[4].cpu().numpy()
    # pdb.set_trace()
    for i in range(prompt_points.shape[1]):
        plotter.add_mesh(
            pv.Sphere(center=prompt_points[0, i, :], radius=prompt_point_radius),
            color=prompt_point_color
        )
    
    ## Plot the 10-click masks
    plotter.subplot(1, 1)
    pred_mask_10 = (seg_logits_prompt_list[9][0].sigmoid().cpu().numpy() > threshold).flatten()
    pc4_colors = colors.copy()
    pc4_colors[pred_mask_10==1] = mask_color
    pc4['RGB'] = pc4_colors
    plotter.add_text(f"10-Click Mask, IoU: {iou_list[9].cpu().numpy().item():.2f}", font_size=10, color='black', position='upper_edge')
    plotter.add_mesh(
        pc4,
        scalars='RGB',
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )

    # Also add the prompt points
    prompt_points = prompt_point_list[9].cpu().numpy()
    for i in range(prompt_points.shape[1]):
        plotter.add_mesh(
            pv.Sphere(center=prompt_points[0, i, :], radius=prompt_point_radius),
            color=prompt_point_color
        )

    # Link the views to synchronize camera movements
    plotter.link_views()
    
    # # Display the plot
    plotter.show()
    # Save screenshot
    # plotter.screenshot(f"save_imgs/{idx}")

    # time.sleep(1)
    
    # Close the plotter
    # plotter.close()

def compute_iou(pred_mask, gt_mask):
    pred_mask = torch.sigmoid(pred_mask) > 0.5
    
    intersection = torch.logical_and(pred_mask, gt_mask).sum(dim=1)
    union = torch.logical_or(pred_mask, gt_mask).sum(dim=1)

    iou = torch.div(intersection, (union + 1e-10))

    if union == 0:
        return 1.0  # If both masks are empty, IoU is 1
    else:
        return intersection / union

def evaluate(args, model, clip_model, val_loader, writer, epoch_num, device, label_dict):
    ####### Perform validation =======================>
    model.train()    
    
    # Freeze the clip backbone
    for param in clip_model.parameters():
        param.requires_grad = False

    text_features_dict = {}
    for dataset_name in label_dict:
        classes = label_dict[dataset_name]
        text_inputs = torch.cat([clip.tokenize(f"segment {classes[c]}") for c in classes]).to(device)
        if args.enable_amp:
            text_features_dict[dataset_name] = clip_model.encode_text(text_inputs) # Get text embeddings for each class in the dataset
        else:
            text_features_dict[dataset_name] = clip_model.encode_text(text_inputs).float()

    with tqdm(total=len(val_loader)) as pbar: 
        for batch_idx_data, sample in enumerate(val_loader):
            # Put the data on device
            sample = all_to_device(sample, device)

            point_offset = []
            point_num_sum = 0
            for i in range(len(sample["masks"])):
                point_num_sum += sample["masks"][i].shape[0]
                point_offset.append(point_num_sum)
            
            sample["point_offset"] = point_offset

            point_offset_list = [0] + sample["point_offset"]
            
            if args.iterative:
                new_click_points = []
                for batch_idx in range(args.batch_size):
                    new_click_points.append(sample['point'][point_offset_list[batch_idx]:point_offset_list[batch_idx+1], :1, :])
                
                sample['point'] = torch.cat(new_click_points, dim=0)
            
            # Run through the model
            with torch.no_grad():
                with autocast(enabled=args.enable_amp):
                    # Run through the backbone first
                    sample = model.module.run_input_encoder(sample)
                    point_features = model.module.run_backbone(sample)

                    # Run through the mask decoder
                    point_offset_list = [0] + sample["point_offset"]
                    offset_list = [0] + sample["offset"].tolist()

                    num_objects_chunk = 1

                    seg_logits_prompt_list_dict = {}
                    prompt_point_list_dict = {}
                    iou_dict = {}
                    for sample_iter in range(args.num_object_points):
                        iou_dict[sample_iter+1] = []
                        seg_logits_prompt_list_dict[sample_iter+1] = []
                        prompt_point_list_dict[sample_iter+1] = []


                    for prompt_idx in range(0, sample['point'].shape[0], num_objects_chunk):
                        end_idx = min(prompt_idx + num_objects_chunk , sample['point'].shape[0])

                        prompt_point_list = []
                        prompt_point_offset_list_new = []
                        masks_new = []
                        for batch_idx in range(args.batch_size):
                            data_dict = sample.copy()
                            
                            # Take the query prompt point and make batch
                            prompt_point_list.append(sample['point'][point_offset_list[batch_idx]:point_offset_list[batch_idx+1], :, :][prompt_idx:end_idx])
                            prompt_point_offset_list_new.append(end_idx - prompt_idx)
                            masks_new.append(sample['masks'][batch_idx][prompt_idx:end_idx])

                        data_dict['point'] = torch.cat(prompt_point_list, dim=0)
                        data_dict['point_offset'] = prompt_point_offset_list_new
                        data_dict['masks'] = masks_new

                        if args.iterative:
                            for sample_iter in range(args.num_object_points):
                                # print("working on sample iter: ", sample_iter)
                                seg_logits_prompt, text_out_prompt, iou_out_prompt, aux_seg_logits_prompt, loss_weights_prompt, _, _ = model.module.run_mask_decoder(point_features, data_dict)
                                # pdb.set_trace()
                                seg_logits_prompt_list_dict[sample_iter+1].append(seg_logits_prompt)
                                prompt_point_list_dict[sample_iter+1].append(data_dict['point'])
                                iou_dict[sample_iter+1].append(compute_iou(seg_logits_prompt[0], data_dict['masks'][0]))

                                if sample_iter < args.num_object_points - 1:
                                    # Based on the ground-truth masks -> Sample new clicks from the error region.
                                    new_clicks = random_sample_clicks(seg_logits_prompt, data_dict['masks'], data_dict['coord'], data_dict['offset'], model.module.mask_threshold)
                                    data_dict['point'] = torch.cat((data_dict['point'], new_clicks), dim=1) # add the new clicks
                        
                        else:
                            seg_logits_prompt, text_out_prompt, iou_out_prompt, aux_seg_logits_prompt, loss_weights_prompt = model.module(data_dict)

                    # Plot the point cloud and all the masks for all the objects in the current batch.
                    plot_pointcloud_all(args, sample, seg_logits_prompt_list_dict, prompt_point_list_dict, iou_dict, idx=f"{batch_idx_data}_{prompt_idx}")

            # Update the tqdm bar
            pbar.set_description(f"Mem: {torch.cuda.max_memory_allocated(device=device)/(1073741824):.2f}")
            pbar.update(1)

def setup_dataloaders(args, config_list):
    assert len(args.stage) == len(config_list)
    
    label_dict = {}
    train_dataset_list = []
    val_dataset_list = []
    for stage, config in zip(args.stage, config_list):
        train_dataset_i = build_dataset_single_mask(config, stage, split="train", num_prompt_points=args.num_prompt_points, 
                            num_object_points=args.num_object_points, overfit=args.overfit, use_random_clicks=args.use_random_clicks)
        val_dataset_i = build_dataset_single_mask(config, stage, split="val", num_prompt_points=args.num_prompt_points, 
                            num_object_points=args.num_object_points, overfit=args.overfit, use_random_clicks=args.use_random_clicks)

        train_dataset_list.append(train_dataset_i)
        val_dataset_list.append(val_dataset_i)
        
        if stage == "kitti":
            label_dict["SemanticKITTI"] = train_dataset_i.class_labels()
        elif stage == "nuscenes":
            label_dict["NuScenes"] = train_dataset_i.class_labels()
        elif stage == "pandaset":
            label_dict["Pandaset"] = train_dataset_i.class_labels()
        elif stage == "scannet":
            label_dict["ScanNet"] = train_dataset_i.class_labels()
        elif stage == "scannet20":
            label_dict["ScanNet"] = train_dataset_i.class_labels()
        elif stage == "kitti360":
            label_dict["KITTI-360"] = train_dataset_i.class_labels()
        elif stage == "s3dis":
            label_dict["S3DIS"] = train_dataset_i.class_labels()
        elif stage == "scannetpp":
            label_dict["ScanNetPP"] = train_dataset_i.class_labels()
        elif stage == "scannet-block":
            label_dict["ScanNetBlock"] = train_dataset_i.class_labels()
        elif stage == "partnet":
            label_dict["PartNet"] = train_dataset_i.class_labels()
        elif stage == "stpls3d":
            label_dict["STPLS3D"] = train_dataset_i.class_labels()


    train_dataset = MultiDataset(train_dataset_list)
    val_dataset = MultiDataset(val_dataset_list)

    if args.local_rank == 0:
        for i, stage in enumerate(args.stage):
            logging.info(f'Number of training samples in {stage}: {len(train_dataset_list[i])}')
            logging.info(f'Number of validation samples in {stage}: {len(val_dataset_list[i])}')

        logging.info(f"Total training samples: {len(train_dataset)}")
        logging.info(f"Total validation samples: {len(val_dataset)}")
    
    # If using distributed, we need to intialize distributed sampler
    if args.distributed:
        _, world_size = get_dist_info()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=args.local_rank)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=args.local_rank)
    else:
        train_sampler = None
        val_sampler = None

    # Initialize the dataloader
    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler, collate_fn=collate_utils.collate_fn)

    # Load validation dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            shuffle=False, num_workers=args.num_workers,
                                            pin_memory=True, sampler=val_sampler,
                                            collate_fn=collate_utils.collate_fn)

    return train_loader, val_loader, train_sampler, label_dict

def get_config(args):
    config_list = []
    for stage in args.stage:
        if stage == "kitti":
            config_list.append(semantic_kitti_new)
        elif stage == "nuscenes":
            config_list.append(nuscenes_new)
        elif stage == "pandaset":
            config_list.append(pandaset)
        elif stage == "scannet":
            config_list.append(scannet)
        elif stage == "scannet20":
            config_list.append(scannet)
        elif stage == "s3dis":
            config_list.append(s3dis)
        elif stage == "kitti360":
            config_list.append(kitti360)
        elif stage == "scanrefer":
            config_list.append(scanrefer)
        elif stage == "scannetpp":
            config_list.append(scannetpp)
        elif stage == "scannet-block":
            config_list.append(scannet_block)
        elif stage == "partnet":
            config_list.append(partnet)
        elif stage == "stpls3d":
            config_list.append(stpls3d)
        else:
            raise ValueError(f"Config file not found for stage: {stage}")

    return config_list

def main(args):
    #TODO: Modify the code to read args from a config file as well and save that config file in logs
    curr_date_time = datetime.now().strftime('%y%m%d_%H%M%S')
    
    if args.dev:
        args.logdir = "dev_logs/"
        args.checkpoint_dir = "dev_checkpoints/"
    
    print(f"Saving logs in {args.logdir}/{args.exp_name}_{curr_date_time}")
    
    # Modify the current experiment name with current date and time
    args.exp_name = f"{curr_date_time}_{args.exp_name}"

    if args.local_rank == 0:
        # Intialize the tensorboard summary writer
        writer = SummaryWriter(f"{args.logdir}/{args.exp_name}/Tensorboard/")
        # Intialize the logger file
        logger = create_logger(name = f'{args.logdir}/{args.exp_name}/log.txt')
    else:
        writer = None
        logger = None

    if args.local_rank == 0:
        # Log available GPUs
        logging.info(f'Available GPUs: {torch.cuda.device_count()}')

        # Log current settings
        logging.info("The current settings are as follows:")
        for arg in vars(args):
            logging.info(f"{arg}: {getattr(args, arg)}")

    # Params for distributed training
    # torch.backends.cudnn.benchmark = True
    if args.distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model #TODO: Fix this # Temporarily fixed
    model = SNAP(num_points=args.num_prompt_points, num_merge_blocks=args.num_merge, use_pdnorm=args.use_pdnorm, 
                    use_aux_loss=args.use_aux_loss, use_localized_loss=args.use_localized_loss).to(device)

    # If resuming from an existing checkpoint
    if args.resume:
        model = load_model_weights(model, args.resume)

    # Generate text embeddings for each class in the dataset
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # If using distributed, create a distributed model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
            )

    # Log the total available training and total parameters
    if args.local_rank == 0:
        logging.info(f'Number of trainable params: {sum(p.numel() for p in model.parameters())}')
    
    # Load config
    config_list = get_config(args)
    
    # Load dataloaders
    train_loader, val_loader, train_sampler, label_dict = setup_dataloaders(args, config_list)

    # evaluate(args, model, clip_model, val_loader, writer, device, label_dict)
    evaluate(args, model, clip_model, val_loader, writer, 0, device, label_dict)

    # Close the summary writer
    if args.local_rank == 0:
        writer.close()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    main(args)