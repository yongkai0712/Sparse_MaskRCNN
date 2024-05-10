import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform

from SparseRoIHead import SparseRoIHeads
from sparse_transform import GeneralizedRCNNTransform_s

# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(
#         in_features_mask, hidden_layer, num_classes
        
#     )
#     # min_size = 800
#     # max_size = 1360


#     # image_mean = [0.485, 0.456, 0.406]

#     # image_std = [0.229, 0.224, 0.225]
#     # model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
#     return model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_sparse_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    outchannel = model.backbone.out_channels
    sparse_roi_heads = SparseRoIHeads(
        box_roi_pool = model.roi_heads.box_roi_pool,
        box_head = model.roi_heads.box_head,
        box_predictor = FastRCNNPredictor(in_features, num_classes),
        fg_iou_thresh = 0.5,
        bg_iou_thresh = 0.5,
        batch_size_per_image = 512,
        positive_fraction = 0.25,
        bbox_reg_weights = None,
        score_thresh = 0.05,
        nms_thresh = 0.5,
        detections_per_img = 100,
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2),
        mask_head = MaskRCNNHeads(outchannel, (256,256,256,256), 1),
        mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
    )

    min_size = 800
    max_size = 1360

    
    image_mean = [0.485, 0.456, 0.406]

    image_std = [0.229, 0.224, 0.225]
    model.transform = GeneralizedRCNNTransform_s(min_size, max_size, image_mean, image_std)

    model.roi_heads = sparse_roi_heads

    
    return model
