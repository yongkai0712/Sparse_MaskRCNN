# 假设您已经有了一个名为 val_dataset 的验证数据集

from torchvision.utils import draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt
import torch

# 在训练循环外部定义一个函数来执行验证和可视化
def validate_and_visualize(model, val_loader, device, num_images=5):
    model.eval()  # 设置模型为评估模式
    images, outputs = [], []
    with torch.no_grad():  # 在这个阶段不计算梯度
        for i, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            output = model(images)
            
            # 可视化前几个图像的预测结果
            if i < num_images:
                for img, pred in zip(images, output):
                    print(pred)
                    pred_masks = pred['masks'] > 0.5  # 假设预测掩码的阈值是0.5
                    vis_image = draw_segmentation_masks(img.cpu(), pred_masks, alpha=0.6)
                    plt.imshow(vis_image.permute(1, 2, 0))
                    plt.axis('off')
                    plt.show()
            else:
                break

    model.train()  # 将模型恢复到训练模式

# 在您的训练循环中，在每个epoch结束时调用这个函数
# 例如：
# for epoch in range(num_epochs):
#     train_one_epoch(...)  # 您的训练函数
#     validate_and_visualize(model, val_loader, device)
