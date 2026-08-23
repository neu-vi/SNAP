import numpy as np
import pdb
import torch
from uuid import uuid4
import torch.nn.functional as F
from copy import deepcopy

opt_overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)

def evaluate_matches_classwise(matches, CLASS_LABELS):
    overlaps = opt_overlaps
    min_region_sizes = [100]
    dist_threshes = [float('inf')]
    dist_confs = [-float('inf')]
    
    # results: class x overlap
    ap = np.zeros( (len(dist_threshes) , len(CLASS_LABELS) , len(overlaps)) , float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        # pdb.set_trace()
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true  = np.ones ( len(gt_instances) )
                    cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                    cur_match = np.zeros( len(gt_instances) , dtype=bool)
                    # collect matches
                    for (gti,gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max( cur_score[gti] , confidence )
                                    min_score = min( cur_score[gti] , confidence )
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true  = np.append(cur_true,0)
                                    cur_score = np.append(cur_score,min_score)
                                    cur_match = np.append(cur_match,True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true  = cur_true [ cur_match==True ]
                    cur_score = cur_score[ cur_match==True ]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore)/pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true,0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score,confidence)

                    # append to overall results
                    y_true  = np.append(y_true,cur_true)
                    y_score = np.append(y_score,cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort      = np.argsort(y_score)
                    y_score_sorted      = y_score[score_arg_sort]
                    y_true_sorted       = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples      = len(y_score_sorted)
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples =  y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision         = np.zeros(num_prec_recall)
                    recall            = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                    # deal with remaining
                    for idx_res,idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores-1]
                        tp = num_true_examples - cumsum
                        fp = num_examples      - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idx_res] = p
                        recall   [idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di,li,oi] = ap_current
    return ap

def evaluate_matches(
    results_file, len_gt_instances: int
):
    overlaps = opt_overlaps
    cur_true = np.ones(len_gt_instances)
    cur_score = np.ones(len_gt_instances) * (-float("inf"))
    cur_match = np.zeros(len_gt_instances, dtype=bool)
    ap = np.zeros((1, 1, len(overlaps)), float)
    for oi, overlap_th in enumerate(overlaps):
        hard_false_negatives = 0
        positives = 0
        cur_true = np.ones(len_gt_instances)
        cur_score = np.ones(len_gt_instances) * (-float("inf"))
        cur_match = np.zeros(len_gt_instances, dtype=bool)

        y_true = np.empty(0)
        y_score = np.empty(0)
        gti = 0
        for idx, (score, iou) in enumerate(results_file):
            if iou > overlap_th:
                cur_match[gti] = True
                cur_score[gti] = iou
                positives += 1
            else:
                hard_false_negatives += 1
                cur_score[gti] = iou
            gti += 1

        # remove non-matched ground truth instances
        cur_true = cur_true[cur_match]
        cur_score = cur_score[cur_match]

        # append to overall results
        y_true = np.append(y_true, cur_true)
        y_score = np.append(y_score, cur_score)

        # sorting and cumsum
        score_arg_sort = np.argsort(y_score)
        y_score_sorted = y_score[score_arg_sort]
        y_true_sorted = y_true[score_arg_sort]
        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

        # unique thresholds
        (thresholds, unique_indices) = np.unique(
            y_score_sorted, return_index=True
        )
        num_prec_recall = len(unique_indices) + 1

        # prepare precision recall
        num_examples = len(y_score_sorted)

        if y_true_sorted_cumsum.size > 0:
            num_true_examples = y_true_sorted_cumsum[-1]
        else:
            num_true_examples = 0

        precision = np.zeros(num_prec_recall)
        recall = np.zeros(num_prec_recall)

        # deal with the first point
        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)

        # deal with remaining
        for idx_res, idx_scores in enumerate(unique_indices):
            cumsum = y_true_sorted_cumsum[idx_scores - 1]
            tp = num_true_examples - cumsum
            fp = num_examples - idx_scores - tp
            fn = cumsum + hard_false_negatives
            p = float(tp) / (tp + fp)
            r = float(tp) / (tp + fn)

            precision[idx_res] = p
            recall[idx_res] = r

        # first point in curve is artificial
        precision[-1] = 1.0
        recall[-1] = 0.0

        # compute average of precision-recall curve
        recall_for_conv = np.copy(recall)
        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
        recall_for_conv = np.append(recall_for_conv, 0.0)

        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")

        # integrate is now simply a dot product
        ap_current = np.dot(precision, stepWidths)

        ap[0, 0, oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt_overlaps, 0.50))
    o25 = np.where(np.isclose(opt_overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt_overlaps, 0.25)))

    avg_dict = {}
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}

    for (li, label_name) in enumerate(["agnostic"]):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(
            aps[d_inf, li, oAllBut25]
        )
        avg_dict["classes"][label_name]["ap50%"] = np.average(
            aps[d_inf, li, o50]
        )
        avg_dict["classes"][label_name]["ap25%"] = np.average(
            aps[d_inf, li, o25]
        )
    return avg_dict

def compute_ap_classewise(total_ap_info_click, exclude_wall_floor=False):
    
    class_scores = {}
    for label in total_ap_info_click:
        num_instances = len(total_ap_info_click[label])
        ap_scores = evaluate_matches(total_ap_info_click[label], num_instances)
        avgs = compute_averages(ap_scores)
        class_scores[label] = avgs

    if exclude_wall_floor:
        # Compute the mean of the class-wise AP results excluding wall and floor
        mean_ap = np.mean([class_scores[label]["all_ap"] for label in class_scores if label not in [0, 1]])
        mean_ap_50 = np.mean([class_scores[label]["all_ap_50%"] for label in class_scores if label not in [0, 1]])
        mean_ap_25 = np.mean([class_scores[label]["all_ap_25%"] for label in class_scores if label not in [0, 1]])
    else:
        # Compute the mean of the class-wise AP results
        mean_ap = np.mean([class_scores[label]["all_ap"] for label in class_scores])
        mean_ap_50 = np.mean([class_scores[label]["all_ap_50%"] for label in class_scores])
        mean_ap_25 = np.mean([class_scores[label]["all_ap_25%"] for label in class_scores])

    return class_scores, mean_ap, mean_ap_50, mean_ap_25
        
def compute_ap_instancewise(total_ap_info_click):
    # Aggrete the results for all instances for each label
    instance_data = []
    for label in total_ap_info_click:
        for (score, iou) in total_ap_info_click[label]:
            instance_data.append((score, iou))

    # Compute the AP for the instance-wise data
    ap_scores = evaluate_matches(instance_data, len(instance_data))
    avgs = compute_averages(ap_scores)

    return avgs

def compute_pq_metrics(num_classes, pred_masks, gt_masks, gt_labels, clip_token, text_token, mask_threshold=0.5, refer_labels=False, return_confusion=False):
    """
    Compute PQ, SQ, and RQ metrics.

    pred_masks: List of predicted masks (list of tensors)
    pred_labels: List of predicted labels (list of tensors)
    gt_masks: List of ground truth masks (list of tensors)
    gt_labels: List of ground truth labels (list of tensors)
    """
    # TP: predicted mask that match a gt mask with IoU > 0.5 and correct labels
    # FP: predicted mask that have IoU < 0.5 or have incorrect labels
    # FN: equals to 0 in the case of promptable segmentation

    # RQ = TP / (TP + 0.5 * FP + 0.5 * FN) = TP / (TP + FP)
    # SQ = Sum(IoU) / (TP)
    # PQ = RQ * SQ

    batch_size = len(pred_masks)

    # Normalize the CLIP tokens
    clip_token = F.normalize(clip_token, p=2, dim=1)

    # Make a dict to store TP, FP and iou_sum class wise
    metrics_dict = {}
    for i in range(num_classes):
        metrics_dict[i] = {"TP": torch.tensor(0).to(clip_token.device), "FP": torch.tensor(0).to(clip_token.device), "iou_sum": torch.tensor(0.0).to(clip_token.device)}
    
    # Initialize confusion matrix if requested
    if return_confusion:
        # +1 for background/unmatched cases
        confusion_matrix = torch.zeros((num_classes, num_classes+1), dtype=torch.long, device=clip_token.device)

    for i in range(batch_size):
        pred_i = pred_masks[i] # Shape -> (M, N)
        gt_i = gt_masks[i] # Shape -> (M, N)
        gt_labels_i = gt_labels[i]  # [M]

        # Normalize predicted text embeddings
        pred_text_token_i = F.normalize(text_token[i], p=2, dim=1)
        
        # Compute similarity scores between predicted tokens and class tokens
        text_logits = pred_text_token_i @ clip_token.T
        
        # Predicted labels for each predicted mask
        pred_labels = torch.argmax(text_logits, dim=1)
        # pred_mask = F.sigmoid(pred_i) > mask_threshold

        # # Compute IoU for each predicted mask
        # intersection = torch.logical_and(pred_mask, gt_i).sum(dim=1)
        # union = torch.logical_or(pred_mask, gt_i).sum(dim=1)

        # iou = torch.div(intersection, (union + 1e-10))
        # iou_mask = iou > mask_threshold

        # Compute TP, FP, iou_sum for each class
        for idx in range(pred_i.shape[0]):
            pred_i_element = pred_i[idx, :]
            gt_i_element = gt_i[idx, :]
            pred_i_element_mask = F.sigmoid(pred_i_element) > mask_threshold

            # Compute IoU
            intersection = torch.logical_and(pred_i_element_mask, gt_i_element).sum()
            union = torch.logical_or(pred_i_element_mask, gt_i_element).sum()
            iou = intersection / (union + 1e-10)

            # Update confusion matrix if requested
            if return_confusion:
                if iou > mask_threshold:
                    # Detected object - record actual vs predicted class
                    confusion_matrix[gt_labels_i[idx].item(), pred_labels[idx].item()] += 1
                else:
                    # Failed to detect object (false negative)
                    confusion_matrix[gt_labels_i[idx].item(), num_classes] += 1
                    # confusion_matrix[gt_labels_i[idx].item(), pred_labels[idx].item()] += 1

            # Compute TP, FP, iou_sum
            if refer_labels:
                if iou > mask_threshold and pred_labels[idx] == gt_labels_i[idx]:
                    metrics_dict[gt_labels_i[idx].item()]["TP"] += 1
                    metrics_dict[gt_labels_i[idx].item()]["iou_sum"] += iou
                else:
                    metrics_dict[gt_labels_i[idx].item()]["FP"] += 1

            else:
                if iou > mask_threshold:
                    metrics_dict[gt_labels_i[idx].item()]["TP"] += 1
                    metrics_dict[gt_labels_i[idx].item()]["iou_sum"] += iou
                else:
                    metrics_dict[gt_labels_i[idx].item()]["FP"] += 1
            
    if return_confusion:
        return metrics_dict, confusion_matrix
    else:
        return metrics_dict, None

def get_labels(clip_token, text_token):
    batch_size = len(text_token)
    # Normalize the CLIP tokens
    clip_token = F.normalize(clip_token, p=2, dim=1)
    out_list = []
    for i in range(batch_size):
        # Normalize predicted text embeddings
        pred_text_token_i = F.normalize(text_token[i], p=2, dim=1)
        
        # Compute similarity scores between predicted tokens and class tokens
        text_logits = pred_text_token_i @ clip_token.T
        
        # Predicted labels for each predicted mask
        pred_labels = torch.argmax(text_logits, dim=1)

        out_list.append(pred_labels)

    return out_list

def compute_pq_metrics_by_matching(
    pred_masks: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.5,
    ignore_duplicates: bool = False,
    class_agnostic: bool = True
):
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

    Args:
        pred_masks (torch.Tensor): A boolean tensor of predicted masks, shape (Q, N),
                                   where Q is the number of predicted instances and N is the number of points.
        pred_labels (torch.Tensor): A long tensor of predicted class labels, shape (Q,).
        gt_masks (torch.Tensor): A boolean tensor of ground truth masks, shape (M, N),
                                 where M is the number of ground truth instances.
        gt_labels (torch.Tensor): A long tensor of ground truth class labels, shape (M,).
        num_classes (int): The total number of classes in the dataset.
        iou_threshold (float): The IoU threshold to consider a match a True Positive.

    Returns:
        dict: A dictionary containing the overall PQ, SQ, RQ, and per-class metrics.
    """
    if class_agnostic:
        # To make the logic class-agnostic, we can simply force all labels to be the same (e.g., 0)
        # and evaluate on a single universal class.
        pred_labels = torch.zeros_like(pred_labels)
        gt_labels = torch.zeros_like(gt_labels)
        num_classes = 1  # We will only loop once for our universal class.

    # --- Data Structures to Store Results ---
    # We will store TP, FP, FN, and the sum of IoUs for each class
    per_class_stats = {
        i: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0, 'tp_mask_points': 0}
        for i in range(num_classes)
    }

    # --- Step 1: Match Predictions to Ground Truth for each Class ---
    for class_id in range(num_classes):
        # 1a. Get all predictions and ground truths for the current class
        pred_indices_class = torch.where(pred_labels == class_id)[0]
        gt_indices_class = torch.where(gt_labels == class_id)[0]

        # If there are no predictions or no ground truths for this class, we can skip
        if len(pred_indices_class) == 0:
            per_class_stats[class_id]['fn'] += len(gt_indices_class)
            continue
        if len(gt_indices_class) == 0:
            per_class_stats[class_id]['fp'] += len(pred_indices_class)
            continue

        pred_masks_class = pred_masks[pred_indices_class]
        gt_masks_class = gt_masks[gt_indices_class]

        # 1b. Compute the IoU matrix between all pairs of predictions and ground truths
        intersection = torch.matmul(pred_masks_class.float(), gt_masks_class.float().T)
        pred_areas = pred_masks_class.sum(dim=1).unsqueeze(1)
        gt_areas = gt_masks_class.sum(dim=1).unsqueeze(0)
        union = pred_areas + gt_areas - intersection
        
        # Handle division by zero for empty unions
        iou_matrix = torch.where(union > 0, intersection / union, torch.tensor(0.0, device=pred_masks.device))

        # 1c. Find potential matches above the IoU threshold
        potential_matches = iou_matrix > iou_threshold
        
        # 1d. Greedy matching: Find the best match for each ground truth object
        # This implementation uses a simple greedy approach. A more optimal one could use the Hungarian algorithm,
        # but this is standard and effective for PQ.
        gt_matched = torch.zeros(gt_masks_class.shape[0], dtype=torch.bool, device=pred_masks.device)
        pred_matched = torch.zeros(pred_masks_class.shape[0], dtype=torch.bool, device=pred_masks.device)

        # Sort potential matches by IoU score to be greedy
        sorted_iou_indices = torch.stack(torch.where(potential_matches)).T
        if sorted_iou_indices.shape[0] > 0:
            sorted_ious = iou_matrix[sorted_iou_indices[:, 0], sorted_iou_indices[:, 1]]
            sort_order = torch.argsort(sorted_ious, descending=True)
            sorted_iou_indices = sorted_iou_indices[sort_order]

            for pred_idx, gt_idx in sorted_iou_indices:
                if not gt_matched[gt_idx] and not pred_matched[pred_idx]:
                    # This is a True Positive
                    per_class_stats[class_id]['tp'] += 1
                    per_class_stats[class_id]['iou_sum'] += iou_matrix[pred_idx, gt_idx].item()
                    gt_matched[gt_idx] = True
                    pred_matched[pred_idx] = True

                    # Count the number of points in the TP mask
                    # This is useful for debugging or further analysis, but not directly for PQ calculation.
                    per_class_stats[class_id]['tp_mask_points'] += pred_masks_class[pred_idx].sum().item()


        # --- Step 2: Calculate FP and FN for the class ---
        if not ignore_duplicates:
            # Standard PQ: All unmatched predictions are counted as False Positives.
            per_class_stats[class_id]['fp'] += (pred_matched == False).sum().item()
        else:
            # New behavior: Only count FPs that are not duplicates.
            unmatched_pred_indices = torch.where(pred_matched == False)[0]
            for pred_idx in unmatched_pred_indices:
                # Check if this unmatched prediction had any potential match above the threshold.
                # If it did, it was a duplicate of a TP. If not, it was a true error.
                is_duplicate = torch.any(iou_matrix[pred_idx, :] > iou_threshold)
                
                if not is_duplicate:
                    # Only count as FP if it was not a duplicate (i.e., a hallucination or low IoU).
                    per_class_stats[class_id]['fp'] += 1

        # False Negatives are always the ground truth instances that were not matched.
        per_class_stats[class_id]['fn'] += (gt_matched == False).sum().item()

    return per_class_stats

def compute_averages_openvocab(aps, DATASET_NAME=None, CLASS_LABELS=None, HEAD_CATS_SCANNET_200=None, COMMON_CATS_SCANNET_200=None, TAIL_CATS_SCANNET_200=None):
    d_inf = 0
    o50 = np.where(np.isclose(opt_overlaps, 0.5))
    o25 = np.where(np.isclose(opt_overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt_overlaps, 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    
    #pdb.set_trace()
    if DATASET_NAME == 'scannet': # compute average scores for head, common, tail categories
        head_scores = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
        common_scores = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
        tail_scores = {title:[] for title in ['ap', 'ap25%', 'ap50%']}
        
    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])

        if DATASET_NAME == 'scannet':
            if (label_name in HEAD_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    head_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    common_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    tail_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            else:
                raise NotImplementedError(label_name)
            
    if DATASET_NAME=='scannet':
        for score_type in ['ap', 'ap25%', 'ap50%']:
            avg_dict['head_'+score_type] = np.nanmean(head_scores[score_type]) #64, orig 66
            avg_dict['common_'+score_type] = np.nanmean(common_scores[score_type]) #68, orig 68
            avg_dict['tail_'+score_type] = np.nanmean(tail_scores[score_type]) #66, orig 66
    
    return avg_dict

def save_data_openvocab(pred_logits, pred_text_out, clip_text_features, gt_masks, gt_labels, label_dict, dataset_name, scene_id, mask_threshold=0.5):
    scene_data = {}
    # Get the predicted masks, predicted labels, predicted_scores, gt_masks, gt_labels and save them to disk
    for batch_idx in range(len(pred_logits)):
        # Make masks from logits
        pred_i = pred_logits[batch_idx] # Shape: M, N
        pred_i_masks = F.sigmoid(pred_i) > mask_threshold # Shape: M, N

        pred_text_out_i = pred_text_out[batch_idx] # Shape: M, D
        normalized_label_tokens = F.normalize(pred_text_out_i, p=2, dim=1)
        normalized_clip_tokens = F.normalize(clip_text_features, p=2, dim=1)
        
        # Compute the similarity scores
        per_class_similarity_scores = normalized_label_tokens @ normalized_clip_tokens.T # Shape: M, D @ D, C -> M, C
        pred_class_indices = torch.argmax(per_class_similarity_scores, dim=1) # Shape: M
        pred_scores = torch.ones(pred_class_indices.shape[0]) # Shape: M

        # Save the data
        # 1. Convert data to numpy first
        pred_i_masks_np = pred_i_masks.cpu().numpy()
        pred_class_indices_np = pred_class_indices.cpu().numpy()
        pred_scores_np = pred_scores.cpu().numpy()
        gt_masks_np = gt_masks[batch_idx].cpu().numpy()
        gt_labels_np = gt_labels[batch_idx].cpu().numpy()

        # Get unique scene id
        scene_id_new = scene_id + f"_{batch_idx}"

        # Store scene data
        scene_data[scene_id_new] = {
            'pred_masks': pred_i_masks_np,
            'pred_scores': pred_scores_np,
            'pred_classes': pred_class_indices_np,
            'gt_masks': gt_masks_np,
            'gt_labels': gt_labels_np,
        }
    
    return scene_data

def convert_masks_to_gt_ids(gt_masks, gt_classes, index_to_scannet_id):
    """
    Convert ground truth masks to gt_ids format expected by the evaluation code.
    
    Args:
        gt_masks: shape (M, N) - M binary masks for N points
        gt_classes: shape (M,) - class indices (0-19)
    
    Returns:
        gt_ids: shape (N,) - instance ID for each point
    """
    N = gt_masks.shape[1]
    gt_ids = np.zeros(N, dtype=np.int32)
    
    # Process each ground truth mask
    for i, (mask, class_idx) in enumerate(zip(gt_masks, gt_classes)):
        # Convert index to actual ScanNet class ID
        scannet_class_id = index_to_scannet_id[int(class_idx)]
        
        # Create instance ID: class_id * 1000 + instance_number
        instance_id = scannet_class_id * 1000 + (i + 1)
        
        # Assign this instance ID to all points in the mask
        gt_ids[mask.astype(bool)] = instance_id
    
    return gt_ids

def make_pred_info(pred: dict, index_to_scannet_id: dict):
    """
    pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    """
    pred_info = {}
    assert (pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
    
    for i in range(len(pred['pred_classes'])):
        info = {}
        # Convert class index to ScanNet ID
        class_idx = int(pred['pred_classes'][i])
        scannet_class_id = index_to_scannet_id[class_idx]
        
        info["label_id"] = scannet_class_id  # Use ScanNet ID instead of index
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][:, i]
        
        # Generate a unique identifier
        unique_id = str(uuid4())
        info["uuid"] = unique_id
        info["filename"] = unique_id
        pred_info[unique_id] = info
    
    return pred_info

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id     = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"

def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances

def assign_instances_for_scan(scene_data: dict, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids):
    # Create pred dict from scene_data
    pred = {
        'pred_scores': scene_data['pred_scores'],
        'pred_classes': scene_data['pred_classes'],
        'pred_masks': scene_data['pred_masks'].T  # Transpose to get shape (N, M)
    }
    
    pred_info = make_pred_info(pred, index_to_correct_ids)

    # Convert gt_masks to gt_ids format
    gt_ids = convert_masks_to_gt_ids(scene_data['gt_masks'], scene_data['gt_labels'], index_to_correct_ids)

    # get gt instances
    gt_instances = get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    # Set default values for all gt instances
    for label_name in gt_instances:
        for gt_inst in gt_instances[label_name]:
            gt_inst['med_dist'] = 0.0  # Default: no distance filtering
            gt_inst['dist_conf'] = 1.0  # Default: high confidence

    # associate
    gt2pred = deepcopy(gt_instances)
    
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []

    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.isin(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        label_id = int(pred_info[uuid]['label_id'])
        conf = pred_info[uuid]['conf']
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]['mask']
        assert (len(pred_mask) == len(gt_ids))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < 100: # TODO: Fix this (https://github.com/JonasSchult/Mask3D/blob/11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5/benchmark/evaluate_semantic_instance.py#L94C18-L94C47)
            continue  # skip if empty

        pred_instance = {}
        pred_instance['uuid'] = uuid
        pred_instance['filename'] = pred_info[uuid]['filename']
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            # pdb.set_trace()
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            # print(f"For label_id: {label_id} class_name: {label_name}, intersection: {intersection} intersection label: {gt_inst['label_id']}, {ID_TO_LABEL[gt_inst['label_id']]} ")
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def assign_instances_for_scan_interactive(scene_data: dict, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids):
    """
    Assigns instances for a single scan based on a one-to-one correspondence
    between ground truth and predictions, suitable for prompt-based models.

    This function assumes that the i-th prediction mask is the model's output
    for the i-th ground truth instance that was prompted.

    Args:
        scene_data (dict): Contains 'gt_masks', 'gt_labels', 'pred_masks',
                           'pred_scores', 'pred_classes'.
        VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids:
                           Dataset-specific mappings.

    Returns:
        tuple: (gt2pred, pred2gt) dictionaries in the format expected by
               the downstream evaluation function.
    """
    # --- Extract data from the scene dictionary ---
    gt_masks = scene_data['gt_masks']
    gt_labels = scene_data['gt_labels']         # Class indices (e.g., 0-199)
    pred_masks = scene_data['pred_masks']
    pred_scores = scene_data['pred_scores']
    pred_classes = scene_data['pred_classes']   # Class indices (e.g., 0-199)

    num_instances = gt_masks.shape[0]
    assert num_instances == pred_masks.shape[0], "Mismatch between number of GT and Predicted instances"

    # --- Initialize containers for the evaluation script ---
    gt2pred = {label: [] for label in CLASS_LABELS}
    pred2gt = {label: [] for label in CLASS_LABELS}

    # --- Loop through each instance with a one-to-one mapping ---
    for i in range(num_instances):
        # 1. Get GT and Pred info for the current instance
        gt_mask_i = gt_masks[i]
        gt_label_idx = int(gt_labels[i])

        pred_mask_i = pred_masks[i]
        pred_score_i = pred_scores[i]
        pred_class_idx = int(pred_classes[i])

        # 2. Convert class indices to actual dataset IDs (e.g., ScanNet IDs) and then to label names
        gt_scannet_id = index_to_correct_ids.get(gt_label_idx)
        pred_scannet_id = index_to_correct_ids.get(pred_class_idx)

        # Skip if the ground truth label is not part of the evaluated classes
        if gt_scannet_id is None or gt_scannet_id not in ID_TO_LABEL:
            continue
        
        gt_label_name = ID_TO_LABEL[gt_scannet_id]
        
        # The predicted class might be different from GT or invalid
        pred_label_name = ID_TO_LABEL.get(pred_scannet_id) if pred_scannet_id in ID_TO_LABEL else None

        # 3. Calculate IoU and point counts
        intersection = np.count_nonzero(np.logical_and(gt_mask_i, pred_mask_i))
        union = np.count_nonzero(np.logical_or(gt_mask_i, pred_mask_i))
        iou = intersection / union if union > 0 else 0

        gt_vert_count = np.count_nonzero(gt_mask_i)
        pred_vert_count = np.count_nonzero(pred_mask_i)
        
        # 4. Create the prediction instance dictionary to be linked to the GT
        # This data structure is what evaluate_matches_classwise expects
        pred_instance_for_gt = {
            'uuid': str(uuid4()),
            'filename': str(uuid4()), # Used as a 'visited' flag in the evaluator
            'confidence': pred_score_i,
            'vert_count': pred_vert_count,
            'intersection': intersection,
        }

        # 5. Create the ground truth instance dictionary
        gt_instance = {
            'instance_id': gt_scannet_id * 1000 + i, # Create a unique ID for the GT instance
            'label_id': gt_scannet_id,
            'vert_count': gt_vert_count,
            'med_dist': 0.0,      # Default value, not used in this context
            'dist_conf': 1.0,     # Default value, not used in this context
            'matched_pred': [pred_instance_for_gt] # CRITICAL: Directly assign the one-to-one prediction
        }
        
        # Add the GT instance to its corresponding class list
        gt2pred[gt_label_name].append(gt_instance)

        # 6. Create the full prediction instance for the pred2gt dict (for completeness)
        if pred_label_name:
            # Create a minimal GT dictionary to be linked from the prediction
            gt_instance_for_pred = {
                'instance_id': gt_instance['instance_id'],
                'vert_count': gt_vert_count,
                'intersection': intersection,
                'med_dist': 0.0,      # Add default value to prevent KeyError
                'dist_conf': 1.0,     # Add default value to prevent KeyError
            }
            
            pred_instance = {
                'uuid': pred_instance_for_gt['uuid'],
                'filename': pred_instance_for_gt['filename'],
                'pred_id': i,
                'label_id': pred_scannet_id,
                'vert_count': pred_vert_count,
                'confidence': pred_score_i,
                'void_intersection': 0, # Assuming no void labels in this setup
                'matched_gt': [gt_instance_for_pred] # Direct one-to-one match
            }
            pred2gt[pred_label_name].append(pred_instance)

    return gt2pred, pred2gt

def compute_openvocab_metrics(scene_data_dict, dataset_name, label_dict, one2one=True):
    """
    scene_data_dict: Dictionary of per scene pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
    """

    HEAD_CATS_SCANNET_200=None
    COMMON_CATS_SCANNET_200=None
    TAIL_CATS_SCANNET_200=None
    if dataset_name=="scannet":
        VALID_CLASS_IDS = (
            2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
            155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
            488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191)

        CLASS_LABELS = (
        'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
        'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
        'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
        'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
        'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
        'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
        'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
        'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
        'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
        'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
        'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress')

        # Create ID_TO_LABEL and LABEL_TO_ID mappings
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        original_class_mapping = {}
        for i in range(len(VALID_CLASS_IDS)):
            original_class_mapping[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

        HEAD_CATS_SCANNET_200 = set(['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 
                                'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 
                                'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 
                                'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture', 
                                'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet', 
                                'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair', 
                                'wall', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity', 
                                'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel', 
                                'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub', 'desk', 'wardrobe', 
                                'clothes dryer', 'radiator', 'shelf'])

        COMMON_CATS_SCANNET_200 = set(["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", 
                                "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", 
                                "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", 
                                "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", 
                                "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", 
                                "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", 
                                "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", 
                                "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", 
                                "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", 
                                "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"])
                                
        TAIL_CATS_SCANNET_200 = set(["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", 
                                "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", 
                                "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", 
                                "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", 
                                "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", 
                                "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", 
                                "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", 
                                "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", 
                                "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", 
                                "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", 
                                "fire alarm", "power strip", "calendar", "poster", "luggage"])
        
    elif dataset_name=="scannet20":
        # Following OpenINS3D - we remove wall, floor and otherfurniture class from Open-vocab evaluation
        VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])

        CLASS_LABELS = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub')

        # Create ID_TO_LABEL and LABEL_TO_ID mappings
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        original_class_mapping = {}
        for i in range(len(VALID_CLASS_IDS)):
            original_class_mapping[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]
    
    elif dataset_name=="stpls3d": 
        VALID_CLASS_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        CLASS_LABELS = ('building', 'vegetation', 'vehicle', 'truck', 'aircraft', 'militaryVehicle', 
                        'bike', 'motorcycle', 'light pole', 'street sign', 'clutter', 'fence')
        
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "replica":
        CLASS_LABELS = [
            "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", 
            "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth", 
            "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", 
            "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow", 
            "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", 
            "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", 
            "tv-stand", "vase", "vent", "wall-plug", "window", "rug"
        ]
        VALID_CLASS_IDS = np.array([
            3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 
            35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 
            79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98
        ])

        # Create ID_TO_LABEL and LABEL_TO_ID mappings
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        original_class_mapping = {}
        for i in range(len(VALID_CLASS_IDS)):
            original_class_mapping[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "s3disfull":
        CLASS_LABELS = [
            "ceiling", "floor", "wall", "beam", "column", "window", "door", "table",
            "chair", "sofa", "bookcase", "board", "clutter"
        ]
        VALID_CLASS_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "scannetpp":
        CLASS_LABELS = ["wall", "ceiling", "floor", "table", "door", "ceiling lamp", "cabinet", "blinds", "curtain", "chair", "storage cabinet", "office chair", 
                            "bookshelf", "whiteboard", "window", "box", "window frame", "monitor", "shelf", "doorframe", "pipe", "heater", "kitchen cabinet", "sofa", 
                            "windowsill", "bed", "shower wall", "trash can", "book", "plant", "blanket", "tv", "computer tower", "kitchen counter", "refrigerator", 
                            "jacket", "electrical duct", "sink", "bag", "picture", "pillow", "towel", "suitcase", "backpack", "crate", "keyboard", "rack", "toilet", 
                            "paper", "printer", "poster", "painting", "microwave", "board", "shoes", "socket", "bottle", "bucket", "cushion", "basket", "shoe rack", 
                            "telephone", "file folder", "cloth", "blind rail", "laptop", "plant pot", "exhaust fan", "cup", "coat hanger", "light switch", "speaker", 
                            "table lamp", "air vent", "clothes hanger", "kettle", "smoke detector", "container", "power strip", "slippers", "paper bag", "mouse", 
                            "cutting board", "toilet paper", "paper towel", "pot", "clock", "pan", "tap", "jar", "soap dispenser", "binder", "bowl", "tissue box", 
                            "whiteboard eraser", "toilet brush", "spray bottle", "headphones", "stapler", "marker"]

        VALID_CLASS_IDS = np.arange(100)

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "matterport":
        CLASS_LABELS = [
            "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf", "picture", "counter",
            "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "other", "ceiling"
        ]

        VALID_CLASS_IDS = np.arange(len(CLASS_LABELS))

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    matches = {}

    for i, (scene_id,scene_data) in enumerate(scene_data_dict.items()):
        matches[scene_id] = {}
        # print(f"\nWorking on scene_id: {scene_id}")
        # print(f"Total objects in this scene: {scene_data['gt_masks'].shape[0]}")
        # print(f"Total GT labels in this scene: {scene_data['gt_labels'].shape[0]}")
        # # print(f"Labels are as follows: {scene_data['gt_labels']}")
        # for gt_label, pred_label in zip(scene_data['gt_labels'], scene_data['pred_classes']):
        #     print(f"GT label: {gt_label}, GT class: {label_dict[gt_label]} | pred_label: {pred_label}, pred_class: {label_dict[pred_label]}")

        gt2pred, pred2gt = assign_instances_for_scan(scene_data, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids)
        # gt2pred, pred2gt = assign_instances_for_scan_interactive(scene_data, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids)
        # if one2one:
        #     gt2pred, pred2gt = assign_instances_for_scan_interactive(scene_data, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids)
        # else:
        #     gt2pred, pred2gt = assign_instances_for_scan(scene_data, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids)

        matches[scene_id]['gt'] = gt2pred
        matches[scene_id]['pred'] = pred2gt

    ap_results = evaluate_matches_classwise(matches, CLASS_LABELS)
    avgs = compute_averages_openvocab(ap_results, DATASET_NAME=dataset_name, CLASS_LABELS=CLASS_LABELS, HEAD_CATS_SCANNET_200=HEAD_CATS_SCANNET_200, 
                                      COMMON_CATS_SCANNET_200=COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200=TAIL_CATS_SCANNET_200)

    return avgs

def compute_instance_seg_metrics(scene_data_dict, dataset_name, label_dict, one2one=True):
    """
    scene_data_dict: Dictionary of per scene pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
    """

    HEAD_CATS_SCANNET_200=None
    COMMON_CATS_SCANNET_200=None
    TAIL_CATS_SCANNET_200=None
    if dataset_name=="scannet":
        VALID_CLASS_IDS = (
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
            155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
            488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191)

        CLASS_LABELS = (
        'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
        'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
        'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
        'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
        'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
        'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
        'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
        'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
        'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
        'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
        'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress')

        # Create ID_TO_LABEL and LABEL_TO_ID mappings
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        original_class_mapping = {}
        for i in range(len(VALID_CLASS_IDS)):
            original_class_mapping[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

        HEAD_CATS_SCANNET_200 = set(['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 
                                'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 
                                'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 
                                'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture', 
                                'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet', 
                                'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair', 
                                'wall', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity', 
                                'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel', 
                                'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub', 'desk', 'wardrobe', 
                                'clothes dryer', 'radiator', 'shelf'])

        COMMON_CATS_SCANNET_200 = set(["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", 
                                "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", 
                                "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", 
                                "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", 
                                "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", 
                                "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", 
                                "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", 
                                "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", 
                                "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", 
                                "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"])
                                
        TAIL_CATS_SCANNET_200 = set(["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", 
                                "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", 
                                "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", 
                                "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", 
                                "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", 
                                "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", 
                                "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", 
                                "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", 
                                "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", 
                                "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", 
                                "fire alarm", "power strip", "calendar", "poster", "luggage"])
        
    elif dataset_name=="scannet20":
        # Following OpenINS3D - we remove wall, floor and otherfurniture class from Open-vocab evaluation
        VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])

        CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub')

        # Create ID_TO_LABEL and LABEL_TO_ID mappings
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        original_class_mapping = {}
        for i in range(len(VALID_CLASS_IDS)):
            original_class_mapping[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]
    
    elif dataset_name=="stpls3d": 
        VALID_CLASS_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        CLASS_LABELS = ('building', 'vegetation', 'vehicle', 'truck', 'aircraft', 'militaryVehicle', 
                        'bike', 'motorcycle', 'light pole', 'street sign', 'clutter', 'fence')
        
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "replica":
        CLASS_LABELS = [
            "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", 
            "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth", 
            "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", 
            "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow", 
            "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", 
            "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", 
            "tv-stand", "vase", "vent", "wall-plug", "window", "rug"
        ]
        VALID_CLASS_IDS = np.array([
            3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 
            35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 
            79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98
        ])

        # Create ID_TO_LABEL and LABEL_TO_ID mappings
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        original_class_mapping = {}
        for i in range(len(VALID_CLASS_IDS)):
            original_class_mapping[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "s3disfull":
        CLASS_LABELS = [
            "ceiling", "floor", "wall", "beam", "column", "window", "door", "table",
            "chair", "sofa", "bookcase", "board", "clutter"
        ]
        VALID_CLASS_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "scannetpp":
        CLASS_LABELS = ["wall", "ceiling", "floor", "table", "door", "ceiling lamp", "cabinet", "blinds", "curtain", "chair", "storage cabinet", "office chair", 
                            "bookshelf", "whiteboard", "window", "box", "window frame", "monitor", "shelf", "doorframe", "pipe", "heater", "kitchen cabinet", "sofa", 
                            "windowsill", "bed", "shower wall", "trash can", "book", "plant", "blanket", "tv", "computer tower", "kitchen counter", "refrigerator", 
                            "jacket", "electrical duct", "sink", "bag", "picture", "pillow", "towel", "suitcase", "backpack", "crate", "keyboard", "rack", "toilet", 
                            "paper", "printer", "poster", "painting", "microwave", "board", "shoes", "socket", "bottle", "bucket", "cushion", "basket", "shoe rack", 
                            "telephone", "file folder", "cloth", "blind rail", "laptop", "plant pot", "exhaust fan", "cup", "coat hanger", "light switch", "speaker", 
                            "table lamp", "air vent", "clothes hanger", "kettle", "smoke detector", "container", "power strip", "slippers", "paper bag", "mouse", 
                            "cutting board", "toilet paper", "paper towel", "pot", "clock", "pan", "tap", "jar", "soap dispenser", "binder", "bowl", "tissue box", 
                            "whiteboard eraser", "toilet brush", "spray bottle", "headphones", "stapler", "marker"]

        VALID_CLASS_IDS = np.arange(100)

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    elif dataset_name == "matterport":
        CLASS_LABELS = [
            "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf", "picture", "counter",
            "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "other", "ceiling"
        ]

        VALID_CLASS_IDS = np.arange(len(CLASS_LABELS))

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        index_to_correct_ids = {}
        for i in range(len(VALID_CLASS_IDS)):
            index_to_correct_ids[i] = VALID_CLASS_IDS[i]

    matches = {}

    for i, (scene_id,scene_data) in enumerate(scene_data_dict.items()):
        matches[scene_id] = {}
        gt2pred, pred2gt = assign_instances_for_scan(scene_data, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL, index_to_correct_ids)

        matches[scene_id]['gt'] = gt2pred
        matches[scene_id]['pred'] = pred2gt

    ap_results = evaluate_matches_classwise(matches, CLASS_LABELS)
    avgs = compute_averages_openvocab(ap_results, DATASET_NAME=dataset_name, CLASS_LABELS=CLASS_LABELS, HEAD_CATS_SCANNET_200=HEAD_CATS_SCANNET_200, 
                                      COMMON_CATS_SCANNET_200=COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200=TAIL_CATS_SCANNET_200)

    return avgs