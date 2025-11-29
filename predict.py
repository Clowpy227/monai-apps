import os
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureChannelFirstd,
    AsDiscrete,
    Activations,
    SaveImage
)
from monai.data import Dataset, DataLoader, list_data_collate, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, DiceHelper

# 设定测试集图片和标签的路径，图片和标签的文件名应该相同
images_path = "./datasets/MSD08/images_thin/"
labels_path = "./datasets/MSD08/labels_thin_bi/"

# 载入的模型路径
model_path = "./save/models/unet-hepatic_vessels_bi.pth"
output_path = "./save/output/" + model_path[len("./save/models/"):-len(".pth")]
os.makedirs(output_path, exist_ok=True)         # 先创建一个文件夹
saver = SaveImage(output_path, output_dtype="uint8")

# 利用os模块自动获取文件夹中的文件名
if not os.path.exists(images_path):
    exit(0)
filename_list = os.listdir(images_path)

# 设置测试集
test_image_file_list = [images_path + filename_list[i] for i in range(50, 61)]
test_label_file_list = [labels_path + filename_list[i] for i in range(50, 61)]
test_dataset = [{"img": img, "seg": seg} for img, seg in zip(test_image_file_list, test_label_file_list)]

# 数据集载入
test_transform = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img"]),
    ]
)
test_ds = Dataset(data=test_dataset, transform=test_transform)
test_loader = DataLoader(dataset=test_ds, num_workers=0, shuffle=False, batch_size=1, collate_fn=list_data_collate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = DiceMetric()
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# 开始测试
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
metric_score_list = []
with torch.no_grad():
    for batch in test_loader:
        img, seg = batch["img"].to(device), batch["seg"].to(device)
        test_outputs = sliding_window_inference(img, 96, 4, model, overlap=0.5)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        test_seg = decollate_batch(seg)
        metric(test_outputs, test_seg)
        for test_output in test_outputs:
            saver(test_output)
    print("evaluation metric:", metric.aggregate().item())
    metric.reset()
