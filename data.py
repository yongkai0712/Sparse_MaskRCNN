import os

import torch
from pycocotools.coco import COCO
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
import numpy

class CocoSparseDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = read_image(img_path)

        num_objs = len(coco_annotation)
        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        masks = torch.zeros((num_objs, img.shape[-2], img.shape[-1]), dtype=torch.uint8)
        # TODO: polygon 转sparse优化

        for i, ann in enumerate(coco_annotation):
            # COCO的边界框格式为[x_min, y_min, width, height]
            # 转换为[x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = ann["bbox"]
            boxes[i] = torch.tensor([x_min, y_min, x_min + width, y_min + height])
            # seg = numpy.array(ann["segmentation"])
            # ann["segmentation"] = seg
            # print(ann)
            masks[i] = torch.tensor(coco.annToMask(ann))
        labels = torch.tensor(
            [ann["category_id"] for ann in coco_annotation], dtype=torch.int64
        )
        image_id = torch.tensor([img_id])
        area = torch.tensor(
            [ann["area"] for ann in coco_annotation], dtype=torch.float32
        )
        iscrowd = torch.tensor(
            [ann["iscrowd"] for ann in coco_annotation], dtype=torch.int64
        )

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = masks.to_sparse_coo()
        # target["masks"] = target["masks"].to_dense()#TODO:将稀疏转为稠密，为可视化做准备
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = read_image(img_path)

        num_objs = len(coco_annotation)
        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        masks = torch.zeros((num_objs, img.shape[-2], img.shape[-1]), dtype=torch.uint8)
        # TODO: polygon 转sparse优化

        for i, ann in enumerate(coco_annotation):
            # COCO的边界框格式为[x_min, y_min, width, height]
            # 转换为[x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = ann["bbox"]
            boxes[i] = torch.tensor([x_min, y_min, x_min + width, y_min + height])
            masks[i] = torch.tensor(coco.annToMask(ann))
        labels = torch.tensor(
            [ann["category_id"] for ann in coco_annotation], dtype=torch.int64
        )
        image_id = torch.tensor([img_id])
        area = torch.tensor(
            [ann["area"] for ann in coco_annotation], dtype=torch.float32
        )
        iscrowd = torch.tensor(
            [ann["iscrowd"] for ann in coco_annotation], dtype=torch.int64
        )

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    import utils
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    
    data = CocoSparseDataset("/home/vik/cell_1360x1024/train", "/home/vik/cell_1360x1024/annotation/fixed_train.json")
    data_loader = torch.utils.data.DataLoader(
    data,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    collate_fn=utils.collate_fn,
)
    # print(data_loader)

    # print("hello")
    # for item in data_loader:
    #     item[1][0]["masks"] = item[1][0]["masks"].to
    def visualize_annotations(data_loader, out_folder='/home/vik/out_file_sparse'):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    
        for i, data in enumerate(data_loader):
            images, targets = data  # 假设每个data包含images和其对应的targets（标注）
            
            for j, image in enumerate(images):
                fig, ax = plt.subplots(1)
                ax.imshow(image.permute(1, 2, 0))  # 将通道从[C, H, W]变为[H, W, C]以供matplotlib使用

                # targets可能包含boxes, labels等，这里仅绘制boxes作为示例
                for box in targets[j]['boxes']:
                    x1, y1, x2,y2 = box 
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                # 保存图片
                plt.axis('off')
                fig.savefig(os.path.join("/home/vik/out_file_sparse", f'image_{i}_{j}.png'), bbox_inches='tight')
                plt.close(fig)
    visualize_annotations(data_loader)
