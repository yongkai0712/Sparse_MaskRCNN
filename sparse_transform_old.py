import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import pickle
from image_list import ImageList

#--
import time
#--
#TODO:定义打印显存占用函数
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()

    print(f"Allocated memory: {allocated / 1024**3:.2f} GB")
    print(f"Cached memory: {cached / 1024**3:.2f} GB")

def _onnx_paste_mask_in_image(mask, box, im_h, im_w):    #使用到的参数中，mask为稀疏格式，box为列表张量

    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = box[2] - box[0] + one           
    h = box[3] - box[1] + one                      #根据box中参数，生成了两个与mask同等大小的张量，用来将mask准备放入image中
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))       #对mask添加一个维度 batch*c*h*w

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))         

    unpaded_im_mask = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0, unpaded_im_mask.to(dtype=torch.float32), zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0, concat_0, zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit._script_if_tracing
def _onnx_paste_mask_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))

    return res_append


@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[-2:]


@torch.jit.unused
def _fake_cast_onnx(v: Tensor) -> float:
    # ONNX requires a tensor but here we fake its type for JIT.
    return v


def mask_interpolate(mask,old_bbox,bbox): #TODO:此处两个循环经过修改之后可以将循环合并
    sparse_matrices_1 = []
    sparse_matrices_2 = []
    N,H,W = mask.size()     #此处应为进行变换后的图片尺寸
    # print(old_bbox)
    # print(bbox)

    j = len(old_bbox)
    # print(old_bbox)
    for i in range(j):
        box = old_bbox[i]
        # print(box)
        # print(i)
        box = box.to(torch.device("cuda"))
        x1, y1, x2, y2 = box
        # 第一个稀疏矩阵
        rows_1 = torch.arange(x1.item(), x2.item()).to(torch.device("cuda"))
        rows_1 = rows_1 - x1
        cols_1 = torch.arange(x1, x2)
        cols_1 = cols_1.to(torch.device("cuda"))
        values_1 = torch.ones(len(cols_1))
        values_1 = values_1.to(torch.device("cuda"))
        
        # print(type(tuple(torch.vstack((rows_1, cols_1).tolist))))
        # print(type(tuple(values_1.tolist())))
        # print(type((x2-x1,W)))
        sparse_matrix_1 = torch.sparse_coo_tensor(torch.vstack((rows_1, cols_1)), values_1, size=(int(x2 - x1),int(W),))
        # sparse_matrix_1 = torch.sparse_coo_tensor((torch.vstack((rows_1, cols_1))), values_1, (x2 - x1, W))
        sparse_matrices_1.append(sparse_matrix_1)
        
        
        # print(sparse_matrix_1)
        # 第二个稀疏矩阵
        rows_2 = torch.arange(y1.item(), y2.item()).to(torch.device("cuda"))
        cols_2 = torch.arange(y1.item(), y2.item()).to(torch.device("cuda"))
        cols_2 = cols_2 - y1
        values_2 = torch.ones(len(cols_2))
        values_2 = values_2.to(torch.device("cuda"))
        sparse_matrix_2 = torch.sparse_coo_tensor(torch.vstack((rows_2, cols_2)), values_2, size=(int(H),int(y2 - y1)))
        sparse_matrices_2.append(sparse_matrix_2)

        
        #print(sparse_matrix_2)
    result_mask = []
    


    for i in range(j):
        current_gt_mask = mask[i]  # 选择当前掩码
        old_box = old_bbox[i]
        new_box = bbox[i]
                        # print(len(old_bbox))
                        # print(old_box)
                        # print(sparse_matrices_1[i])
                        # print(sparse_matrices_2[i])
                        # processed_masks = []  # 用于存储每个处理后的稀疏张量
                        # print(old_box)
                        # print(current_gt_mask)
                        #---
                        
                        
                        # import pickle
                        # with open("old_box.pkl","wb") as f:
                        #     pickle.dump(old_box, f)

                        # with open("current_gt_mask.pkl","wb") as f:
                        #     pickle.dump(current_gt_mask, f)
                        
                        
                        #---
                        # print(i)
                        # 执行左乘和右乘
                        # print(sparse_matrices_1[i].shape)
                        # print(current_gt_mask.shape)
        current_gt_mask = current_gt_mask.to(dtype=torch.float)
        current_gt_mask = current_gt_mask.t()
        left_result_sparse = torch.sparse.mm(sparse_matrices_1[i], current_gt_mask)
        result_sparse = torch.sparse.mm(left_result_sparse, sparse_matrices_2[i])   #TODO:此处更改了
        
        result_dense = result_sparse.to_dense()
        result_dense = result_dense.t()
        
        # print(result_dense)
        
        # resized_mask = roi_align_masks(result_dense,(M,M))[:, 0]  # 调整mask大小
        # processed_mask = result_dense # 简化处理，直接使用原始掩码

        # 上采样缩放以适应 new_box 的大小
        old_box_width = old_box[2] - old_box[0]
        old_box_height = old_box[3] - old_box[1]
        new_box_width = new_box[2] - new_box[0]
        new_box_height = new_box[3] - new_box[1]
        scale_factor_width = (new_box_width / old_box_width).item()  # 提取数值
        scale_factor_height = (new_box_height / old_box_height).item()
        # print(scale_factor_width)

                                # # 上采样 processed_mask
                                # processed_mask_upsampled = F.interpolate(result_dense.unsqueeze(0).unsqueeze(0),
                                #                                          scale_factor=(scale_factor_height, scale_factor_width),
                                #                                          mode='bilinear', align_corners=False).squeeze()


        x1, y1, x2, y2 = map(int, new_box)  # 确保坐标是整数
        target_height = y2 - y1
        target_width = x2 - x1
        
        
        # 裁剪或填充 processed_mask_upsampled 以匹配目标尺寸
        processed_mask_upsampled = F.interpolate(result_dense.unsqueeze(0).unsqueeze(0),
                                                          size=(target_height, target_width),
                                                          mode='bilinear', align_corners=False).squeeze().byte()

                                # 创建一个与原始图像相同大小的空白张量
                                # print(processed_mask_upsampled)
        new_image_height = H*scale_factor_height
        new_image_width = W*scale_factor_width
        full_mask = torch.zeros((int(new_image_height), int(new_image_width)))

                            # 根据 new_box 的坐标将上采样后的掩码放置在正确的位置
                            #x1, y1, x2, y2 = map(int, new_box)  # 确保坐标是整数
        full_mask[y1:y2,x1:x2] = processed_mask_upsampled
                            # print(x2-x1)
                            # print(y2-y1)
                            # 将 full_mask 转换为稀疏张量并存储
        full_mask_sparse = full_mask.to_sparse()
        # print(full_mask_sparse)
        result_mask.append(full_mask_sparse)
        
                            # result_m = torch.sparse.cat(result_mask, dim=0)
                            # print(result_m.shape)
                            # print(full_mask_sparse)
        # TODO：打印出裁剪后mask,观察是否存在空mask
    # print(result_mask)    
    #TODO:将result_mask由list转换为tensor
    
    # all_indices = []
    # all_values = []
    
    # for i,_2d_tensor in enumerate(result_mask):
    #     indices = _2d_tensor.indices()
    #     values = _2d_tensor.values()
        
    #     depth_indices = torch.full((1,indices.shape[1]),i,dtype=torch.long)
    #     extendes_indices = torch.cat((depth_indices,indices),dim=0)
        
    #     all_indices.append(extendes_indices)
    #     all_values.append(values)
        
    # all_indices = torch.cat(all_indices,dim=1)
    # all_values = torch.cat(all_values)
    
    # num_tensors = len(result_mask)
    
    # sparse_result = torch.sparse_coo_tensor(all_indices,all_values,size=(num_tensors,H,W))
        
    return result_mask


def _resize_image_and_masks(old_bbox,
                            bbox,
                            image: Tensor,
                            self_min_size: float,
                            self_max_size: float,
                            target: Optional[Dict[str, Tensor]] = None,
                            fixed_size: Optional[Tuple[int, int]] = None,
                            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:

    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)  # 获取高宽中的最小值
        max_size = torch.max(im_shape).to(dtype=torch.float32)  # 获取高宽中的最大值
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)  # 计算缩放比例

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False)[0]

    if target is None:
         return image, target
    
    # old_bbox = target["boxes"]                                   #修改记录：此处已经对image进行了resize,根据image_shape来进行boxes的变换
    # # 根据图像的缩放比例来缩放bbox
    # bbox = resize_boxes(old_bbox, [h, w], image.shape[-2:])    #标记：调整此处的 bbox放缩顺序，以此作为mask放缩前提
    # target["boxes"] = bbox

#根据old_boxes的原始位置作为裁剪mask的初始依据，将更新后的bbox作为mask进行上采样的依据


    if "masks" in target:            #标记：在此处添加读取box信息，然后利用稀疏的上采样函数，进行大小的变换
        mask = target["masks"]
        print(mask)
        print(old_bbox[0])
        # print(mask.layout)
        # print(mask.shape)

        #---
        # import pickle
        # with open("mask.pkl","wb") as f:
        #     pickle.dump(mask, f)
        #---

        
        mask = mask_interpolate(mask,old_bbox,bbox)
        
        # mask = torch.nn.functional.interpolate(
        #     mask[:, None].float(), size=size, scale_factor=scale_factor, recompute_scale_factor=recompute_scale_factor
        # )[:, 0].byte()  # self.byte() is equivalent to self.to(torch.uint8).
        target["masks"] = mask

    return image, target


def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor

    # refer to: https://github.com/pytorch/vision/issues/5845
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batch, C, H, W]
    # 因为后续的bilinear操作只支持4-D的Tensor
    mask = mask.expand((1, 1, -1, -1))  # -1 means not changing the size of that dimension

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]  # [batch, C, H, W] -> [H, W]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    # 填入原图的目标区域(防止越界)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    # 将resize后的mask填入对应目标区域
    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])]
    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor

    # pytorch官方说对mask进行expand能够略微提升mAP
    # refer to: https://github.com/pytorch/vision/issues/5845
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_mask_in_image_loop(
            masks, boxes, torch.scalar_tensor(im_h, dtype=torch.int64), torch.scalar_tensor(im_w, dtype=torch.int64)
        )[:, None]
    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]  # [num_obj, 1, H, W]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self,
                 min_size: int,
                 max_size: int,
                 image_mean: List[float],
                 image_std: List[float],
                 size_divisible: int = 32,
                 fixed_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size

    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type:(Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息
        Args:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]

        if self.training:
            size = float(self.torch_choice(self.min_size))  # 指定输入图片的最小边长,注意是self.min_size不是min_size
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])    # 指定输入图片的最小边长,注意是self.min_size不是min_size


        if target is None:
            return image, target
        
        old_bbox = target["boxes"]                                   #修改记录：此处已经对image进行了resize,根据image_shape来进行boxes的变换
        # 根据图像的缩放比例来缩放bbox
        # print(old_bbox)
        bbox = resize_boxes(old_bbox, [h, w], image.shape[-2:])    #标记：调整此处的 bbox放缩顺序，以此作为mask放缩前提
        target["boxes"] = bbox

        image, target = _resize_image_and_masks(old_bbox,bbox,image, size, float(self.max_size), target, self.fixed_size)



        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍

        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        # 分别计算一个batch中所有图片中的最大channel, height, width
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                # 将mask映射回原图尺度
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]

        # for item in targets:
        #     for key in item:
        #         print(key)
        # with open("target_1.pkl","wb") as f:
        #     pickle.dump(targets, f)
        # print(str(targets["boxes"]))
        
        for i in range(len(images)):
            image = images[i]
            print(targets[0]["image_id"])
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)  # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)  # 对图像和对应的bboxes缩放到指定范围
            images[i] = image
            
            # print_gpu_memory()
            
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, self.size_divisible)  # 将images打包成一个batch
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        
        memory_before = torch.cuda.memory_allocated()
        # print(memory_before)
        save_image(image, 'output_image.png')
        
        # with open("targets.pkl","wb") as f:
        #     pickle.dump(targets, f)
        
        
        
        # TODO：打印显存占用
        # print("sparse_transform中显存占用")
        # print_gpu_memory()
        
        
        return image_list, targets


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)







