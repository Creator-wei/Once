import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms_class(box_scores, box_preds, nms_config, classwise_acc=None, score_thresh=None, Using_Cls=False,selected_label_cls=None,label_preds= None ):
    #Class_Preds
    src_box_scores = box_scores

    ####################################################################################
    mask_cls=None
    select_cls=None
    max_cls_idx=None
    
    if Using_Cls:
        box_scores = box_scores.squeeze(-1)
        #max_cls_preds,max_cls_idx = torch.max(box_scores,-1)
        cls_threshold_per_class = score_thresh
        #print("Cls_threshold ==")
        #print(cls_threshold_per_class.size())
        cls_th = box_scores.new_zeros(box_scores.shape)
        num_class = len(cls_threshold_per_class)
        for cls_idx in range(num_class):
            class_mask = label_preds = (cls_idx+1)
            cls_th[class_mask] = cls_threshold_per_class[cls_idx]*classwise_acc[cls_idx]
            print("-----------Threshold_hold_cls--------------")
            print(class_mask)
            print(cls_th[class_mask])
            print("------------------END----------------------")
           
    if score_thresh is not None:
        if Using_Cls:
            scores_mask = box_scores >= cls_th
            for label, flag in zip(box_scores.tolist(),scores_mask.tolist()):
                if flag:
                    selected_label_cls[label] += 1
            #selected_label_iou = dict(selected_label_iou)
            print("--------------selected_label_cls--------------")
            print(selected_label_cls)
            print("----------------------------------------------")

        box_scores = box_scores[scores_mask]
        #box_preds
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    #return selected, src_box_scores[selected], mask_cls, select_cls,max_cls_idx
    return selected, src_box_scores[selected], scores_mask, selected_label_cls




def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
