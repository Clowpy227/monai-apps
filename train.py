import os
import torch
import logging
import time
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    AsDiscrete,
    Activations
)
from monai.data import Dataset, DataLoader, list_data_collate, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, DiceHelper
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils.enums import MetricReduction
from torch.optim.lr_scheduler import ReduceLROnPlateau


# 设定训练集图片和标签的路径，图片和标签的文件名应该相同
images_path = "./datasets/MSD08/images_thin/"
labels_path = "./datasets/MSD08/labels_thin_bi/"
models_save_filename = "unet-hepatic_vessels_bi"
checkpoint_save_filename = "checkpoint-"+models_save_filename

# 利用os模块自动获取文件夹中的文件名
if not os.path.exists(images_path):
    exit(0)
filename_list = os.listdir(images_path)

# 设置训练集
train_image_file_list = [images_path + filename_list[i] for i in range(40)]
train_label_file_list = [labels_path + filename_list[i] for i in range(40)]
train_dataset = [{"img": img, "seg": seg} for img, seg in zip(train_image_file_list, train_label_file_list)]
# 设置验证集
validate_image_file_list = [images_path + filename_list[i] for i in range(40, 50)]
validate_label_file_list = [labels_path + filename_list[i] for i in range(40, 50)]
validate_dataset = [{"img": img, "seg": seg} for img, seg in zip(validate_image_file_list, validate_label_file_list)]

# 数据集载入过程
train_transform = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img"]),
        RandRotated(keys=["img", "seg"], range_x=0.78, range_y=0.78, range_z=0.78),
        RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", spatial_size=(96, 96, 96), num_samples=4),
        RandFlipd(keys=["img", "seg"], spatial_axis=0),
        RandFlipd(keys=["img", "seg"], spatial_axis=1),
        RandFlipd(keys=["img", "seg"], spatial_axis=2),
        RandScaleIntensityd(keys=["img"], factors=1.0)
    ]
)
validate_transform = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img"])
    ]
)

# 载入训练集
train_ds = Dataset(train_dataset, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=list_data_collate)
# 载入验证集
validate_ds = Dataset(validate_dataset, transform=validate_transform)
validate_loader = DataLoader(validate_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)

# 网络参数设定
# metric = DiceHelper(include_background=False, reduction=MetricReduction.MEAN)           # 用于多目标分类
metric = DiceMetric()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# post_trans = Compose([Activations(softmax=True, dim=1), AsDiscrete(threshold=0.5)])       # 用于多目标分类
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])        # 用于单目标分类
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)
# loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)          # 用于多目标分类
loss_fn = DiceCELoss(sigmoid=True)    # 用于单目标分割
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", min_lr=0.000001,patience=200, cooldown=100)
epoch = 1
val_interval = 5   # 实施验证间隔
best_metric = -1.0    # 记录最好的Dice值
best_metric_epoch = -1  # 记录最佳的epoch
epoch_loss_values = []  # 损失值统计
metric_values = []      # Dice值统计
best_model_filename = "./save/models/"+models_save_filename+".pth"     # 最佳模型保存的位置
checkpoint_filename = "./save/checkpoints/"+checkpoint_save_filename+".pth"    # 检查点路径
# 日志文件
log_filename = f"./save/logs/train-{models_save_filename}-{time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())}.log"  # 日志文件路径
# log_file = open(log_filename, "w")
# log_file.close()
logging.basicConfig(filename=log_filename,
                    filemode="a",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO,
                    encoding="utf-8")
logger= logging.getLogger(__name__)

# 开始训练
while True:
    start_time = time.time()
    logger.info(f"Epoch: {epoch} / ∞")
    print(f"Epoch: {epoch} / ∞")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_len = len(train_ds) // train_loader.batch_size
        print(f"{step} / {step_len} ， {loss.item():.6f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    scheduler.step(epoch_loss)
    end_time = int(time.time() - start_time)
    logger.info(f"epoch {epoch} average loss: {epoch_loss:.6f}, time: {end_time}s")
    print(f"epoch {epoch} average loss: {epoch_loss:.6f}, time: {end_time}s")

    # 验证
    if epoch % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_value = 0.0
            for val_data in validate_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                val_outputs = sliding_window_inference(val_images, 96, 8, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # print(val_outputs.max())
                # print(val_outputs.shape, val_labels.shape)
                metric(y_pred=val_outputs, y=val_labels)
            metric_value = metric.aggregate().item()
            metric.reset()

            metric_values.append(metric_value)
            if metric_value > best_metric:
                best_metric = metric_value
                best_metric_epoch = epoch
                torch.save(model.state_dict(), best_model_filename)
                logger.info(f"已保存最好的模型至{best_model_filename}")
            logger.info(f"当前Epoch：{epoch}, 当前学习率：{optimizer.param_groups[0]['lr']}, 当前平均Dice值:{metric_value:.6f}，最好的Epoch: {best_metric_epoch}, 最好的平均Dice：:{best_metric:.6f}")
            print(f"当前Epoch：{epoch}, 当前学习率：{optimizer.param_groups[0]['lr']}, 当前平均Dice值:{metric_value:.6f}，最好的Epoch: {best_metric_epoch}, 最好的平均Dice：:{best_metric:.6f}")
    # 每10个epoch保存一次模型断点
    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
            "epoch_losses_values": epoch_loss_values,
            "metric_values": metric_values
        }, checkpoint_filename)
        logger.info(f"已保存{checkpoint_filename}")
        print(f"已保存{checkpoint_filename}")
    # 训练终止条件
    if epoch - best_metric_epoch >= 2000:
        break
    else:
        epoch += 1
logger.info(f"训练结束, best_metric: {best_metric:.6f} at epoch: {best_metric_epoch}")
print(f"训练结束, best_metric: {best_metric:.6f} at epoch: {best_metric_epoch}")
print("Metric Values: ", metric_values)
print("Epoch Loss Values: ", epoch_loss_values)
