from ultralytics.utils.metrics import ConfusionMatrix

cfmtrx = ConfusionMatrix(120, conf=0.25, iou_thres=0.45, task="detect")

process_batch(detections, gt_bboxes, gt_cls)
process_batch(detections, gt_bboxes, gt_cls)