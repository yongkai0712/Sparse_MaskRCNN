import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2 as T

import utils
from data import CocoDataset,CocoSparseDataset
from engine import evaluate, train_one_epoch
from model import get_sparse_model_instance_segmentation,get_model_instance_segmentation
from torchvision.utils import draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt
import torch

#TODO:测i模型输出
def validate_and_visualize(model, val_loader, device, num_images=5):
    model.eval()  # 设置模型为评估模式
    images, outputs = [], []
    with torch.no_grad():  # 在这个阶段不计算梯度
        for i, (images, targets) in enumerate(val_loader):
            
            images = [img.to(device) for img in images]
            output = model(images)
            print(output)
            # 可视化前几个图像的预测结果
            # if i < num_images:
            #     for img, pred in zip(images, output):
            #         print(pred)
            #         pred_masks = pred['masks'] > 0  # 假设预测掩码的阈值是0.5
            #         vis_image = draw_segmentation_masks(img.cpu(), pred_masks, alpha=0.6)
            #         plt.imshow(vis_image.permute(1, 2, 0))
            #         plt.axis('off')
            #         plt.show()
            # else:
            #     break

    model.train()  # 将模型恢复到训练模式


#TODO：定义显存占用显示的函数
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()

    print(f"Allocated memory: {allocated / 1024**3:.2f} GB")
    print(f"Cached memory: {cached / 1024**3:.2f} GB")



def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2

dataset = CocoSparseDataset(
    "/home/vik/cell_1360x1024/train", "cell_1360x1024/annotation/train.json", get_transform(train=False)
)
# dataset = CocoDataset(
#    "/home/vik/cell_1360x1024/train", "/home/vik/cell_1360x1024/annotation/newly_updated_train.json", get_transform(train=True)
# )

val_data = CocoDataset(
    '/home/vik/cell_1360x1024/val','/home/vik/cell_1360x1024/annotation/val.json',
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn,
)

val_lodaer = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    shuffle=0,
    num_workers=0,
    collate_fn=utils.collate_fn,
)

# get the model using our helper function
model = get_sparse_model_instance_segmentation(num_classes)
# model = get_model_instance_segmentation(num_classes)
# move model to the right device
model.to(device)




# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.001)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it just for 2 epochs
num_epochs = 20
#TODO:在训练前载入参数时检测显存占用
# print("开始训练前的显存占用")
# print_gpu_memory()

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    weight_pth = f'/home/vik/sparse/weight/model_weight_{epoch}.pth'
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    
    # update the learning rate
    lr_scheduler.step()
    torch.save(model.state_dict(), weight_pth)
    if epoch>10:
        validate_and_visualize(model,val_lodaer,device)
    
    #TODO:在训练一次后进行显存占用的显示，检测显存占用
    # print("开始训练中的显存占用")
    # print_gpu_memory()
    
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

print("That's it!")


