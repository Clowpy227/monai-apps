import os
import torchmetrics
import torchmetrics.segmentation
from monai.transforms import LoadImage


# 评价指标
dice = torchmetrics.segmentation.GeneralizedDiceScore(num_classes=2, include_background=False, input_format="index")
miou = torchmetrics.segmentation.MeanIoU(num_classes=2, include_background=False, input_format="index")
precision = torchmetrics.Precision(task="binary")

eval_arr = {
    "dice": [],
    "miou": [],
    "precision": []
}

# 目录
pred_path = "./save/output"
mask_path = "./datasets/MSD08/labels_thin_bi"
model_name = "unet-hepatic_vessels_bi"
evaluate_path = os.path.join(pred_path, model_name)
for result_dir in os.listdir(evaluate_path):
    pred_filename = os.path.join(evaluate_path, result_dir, result_dir + "_trans.nii.gz")
    mask_filename = os.path.join(mask_path, result_dir + ".nii.gz")
    pred_result = LoadImage(dtype="uint8", ensure_channel_first=True, simple_keys=True)(pred_filename).long()
    mask_result = LoadImage(dtype="uint8", ensure_channel_first=True, simple_keys=True)(mask_filename).long()

    dice.update(pred_result, mask_result)
    miou.update(pred_result, mask_result)
    precision.update(pred_result, mask_result)
    dice_score = dice.compute().item()
    miou_score = miou.compute().item()
    precision_score = precision.compute().item()
    dice.reset()
    miou.reset()
    print("测试用例：", result_dir, "Dice score: ", dice_score, "MIOU score: ", miou_score, "precision score: ", precision_score)
    eval_arr["dice"].append(dice_score)
    eval_arr["miou"].append(miou_score)
    eval_arr["precision"].append(precision_score)

print("整体情况：", "Dice score:", sum(eval_arr["dice"]) / len(eval_arr["dice"]),"MIOU score:", sum(eval_arr["miou"]) / len(eval_arr["miou"]), "precision score:", sum(eval_arr["precision"]) / len(eval_arr["precision"]))
