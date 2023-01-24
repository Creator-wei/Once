import torch
from .semi_utils import reverse_transform, load_data_to_gpu, construct_pseudo_label
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms_class 

@torch.no_grad()
#######################################################################################
def iou_match_3d_filter(batch_dict, cfgs, iouwise_acc, classwise_acc):
#######################################################################################
    batch_size = batch_dict['batch_size']
    pred_dicts = []
    for index in range(batch_size):
        #index is in different batch
        box_preds = batch_dict['rois'][index]
        iou_preds = batch_dict['roi_ious'][index]
        cls_preds = batch_dict['roi_scores'][index]
        label_preds = batch_dict['roi_labels'][index]
        #False
        #using this to normalized the input
        if not batch_dict['cls_preds_normalized']:
            iou_preds = torch.sigmoid(iou_preds)
            cls_preds = torch.sigmoid(cls_preds)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(iou_preds.size())
        print(cls_preds.size())
        print(iou_preds)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        iou_preds = iou_preds.squeeze(-1)
        ###############################################################################
        max_iou_preds,max_iou_idx = torch.max(iou_preds,-1)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(max_iou_preds)
        print(max_iou_idx)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        ###############################################################################
        # filtered by iou_threshold
        iou_threshold_per_class = cfgs.IOU_SCORE_THRESH
        num_classes = len(iou_threshold_per_class)
        iou_th = iou_preds.new_zeros(iou_preds.shape)
        print("---------------1--------------")
        print("iou_preds--------------------")
        print(iou_preds.size())
        for cls_idx in range(num_classes):
            class_mask = label_preds == (cls_idx + 1)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(class_mask)
            print(label_preds)
            print(cls_idx+1)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            iou_th[class_mask] = iou_threshold_per_class[cls_idx]
            print("----------------for----------------------")
            print(iou_th[class_mask])
            print(iou_th[class_mask].size())
            
            ###############################################################################
            #using to loss
            mask_iou = max_iou_preds.ge(iou_th[class_mask]*(iouwise_acc[max_iou_idx]/(2.-iouwise_acc[max_iou_idx]))).float()
            #using to update iouwise_acc
            select_iou = max_iou_preds.ge(iou_th[class_mask]).long()
            ###############################################################################
        #先筛选可能的框
        iou_mask = iou_preds >= iou_th
        iou_preds = iou_preds[iou_mask]
        box_preds = box_preds[iou_mask]
        cls_preds = cls_preds[iou_mask]
        label_preds = label_preds[iou_mask]
        print("--------------2---------------")
        print("iou_preds--------------------")
        print(iou_preds.size())
        print("cls_preds--------------------")
        print(cls_preds.size())
        #再根据筛选出的框选出可能的目标
        nms_scores = cls_preds # iou_preds
        #Fillited by class_threshhold
        selected, selected_scores, mask_cls, select_cls,max_cls_idx= class_agnostic_nms_class(
            box_scores=nms_scores, box_preds=box_preds,
            nms_config=cfgs.NMS_CONFIG,
            score_thresh=cfgs.CLS_SCORE_THRESH,
            classwise_acc=classwise_acc,
            Using_Cls=True
        )

        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = box_preds[selected]

        # added filtering boxes with size 0
        zero_mask = (final_boxes[:, 3:6] != 0).all(1)
        final_boxes = final_boxes[zero_mask]
        final_labels = final_labels[zero_mask]
        final_scores = final_scores[zero_mask]
        print("-----------------------Final---------------------")
        print(final_boxes.size())
        print(final_labels.size())
        print(final_scores.size())
        print("-------------------------------------------------")
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels,
        }
        pred_dicts.append(record_dict)

    return pred_dicts, select_iou, select_cls, mask_iou, mask_cls, max_iou_idx, max_cls_idx

def iou_match_3d(teacher_model, student_model,
                  ld_teacher_batch_dict, ld_student_batch_dict,
                  ud_teacher_batch_dict, ud_student_batch_dict,
                  cfgs, epoch_id, dist,
                  classwise_acc, iouwise_acc
                 ):
    assert ld_teacher_batch_dict is None # Only generate labels for unlabeled data

    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)
    #Dist == False
    if not dist:
        #############################################################################################################
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
    else:
        #############################################################################################################
        _, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
    #################################################################################################################
    teacher_boxes, select_iou, select_cls, mask_iou, mask_cls, max_iou_idx, max_cls_idx= iou_match_3d_filter(ud_teacher_batch_dict, cfgs.TEACHER,classwise_acc=classwise_acc, iouwise_acc=iouwise_acc)
    #################################################################################################################
    #using reverse_transform to transform weak augment to strong augment for match the Stduent model
    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    gt_boxes = construct_pseudo_label(teacher_boxes)
    mask = (mask_iou + mask_cls)/2
    #ud_student_batch_dict is pseudo_label
    ud_student_batch_dict['gt_boxes'] = gt_boxes
    #Dist == False
    if not dist:
        #supervised
        #############################################################################################################
        _, ld_ret_dict, _, _ = student_model(ld_student_batch_dict)   
        #unsupervised          
        #############################################################################################################
        _, ud_ret_dict, tb_dict, disp_dict = student_model(ud_student_batch_dict, Using_acc=True, mask= mask)
    else:
        (_, ld_ret_dict, _, _), (_, ud_ret_dict, tb_dict, disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    return loss, tb_dict, disp_dict, select_iou, select_cls, max_iou_idx.long(), max_cls_idx.long()
