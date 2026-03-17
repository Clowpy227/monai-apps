import os
import numpy as np
from PIL import Image

# 评价指标
eval_arr = {
    "dice": [],
    "miou": [],
    "precision": []
}

# 目录
pred_path = "./save/output"
mask_path = "./datasets/MSD08/labels_thin_bi_2d"
model_name = "unet2d-hepatic_vessels_bi"
evaluate_path = os.path.join(pred_path, model_name)
for result_dir in os.listdir(evaluate_path):
    tp, tn, fp, fn = 0, 0, 0, 0
    for filename in os.listdir(os.path.join(evaluate_path, result_dir)):
        pred_arr = np.array(Image.open(os.path.join(evaluate_path, result_dir, filename)))
        mask_arr = np.array(Image.open(os.path.join(mask_path, result_dir, filename[:-len("_trans.png")] + ".png")))
        eval_arr_ = pred_arr + 2 * mask_arr
        tp += np.sum(eval_arr_ == 3)
        tn += np.sum(eval_arr_ == 0)
        fp += np.sum(eval_arr_ == 1)
        fn += np.sum(eval_arr_ == 2)
    dice_score = 2 * tp / (2 * tp + fp + fn)
    miou_score = tp / (tp + fp + fn)
    precision_score = tp / (tp + fp)
    print("测试用例：", result_dir, "Dice score: ", dice_score, "MIOU score: ", miou_score, "precision score: ", precision_score)
    eval_arr["dice"].append(dice_score)
    eval_arr["miou"].append(miou_score)
    eval_arr["precision"].append(precision_score)

print("整体情况：", "Dice score:", sum(eval_arr["dice"]) / len(eval_arr["dice"]),"MIOU score:", sum(eval_arr["miou"]) / len(eval_arr["miou"]), "precision score:", sum(eval_arr["precision"]) / len(eval_arr["precision"]))
