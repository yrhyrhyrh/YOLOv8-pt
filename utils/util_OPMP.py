import copy
import math
import random
import time

import numpy
import torch
import torchvision
from torch.nn.functional import cross_entropy, one_hot


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, shape2[1])  # x1
    coords[:, 1].clamp_(0, shape2[0])  # y1
    coords[:, 2].clamp_(0, shape2[1])  # x2
    coords[:, 3].clamp_(0, shape2[0])  # y2
    return coords


def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    # x = fpn output = [torch.Size([16, 65, 80, 80]), torch.Size([16, 65, 40, 40]), torch.Size([16, 65, 20, 20])]
    assert x is not None
    anchor_points, stride_tensor = [], []
    # strides = tensor([ 8., 16., 32.])
    for i, stride in enumerate(strides):
        # i=0: h,w = 80,80  i=1: h,w = 40,40  i=2: h,w = 20,20
        _, _, h, w = x[i].shape
        # i=0: sx and sy = [0.5 to 79.5 with step of 1]
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        # i=0: stack 80 of sx to become 80x80
        sy, sx = torch.meshgrid(sy, sx)
        # combine sx and sy into 2 columns
        # i=0: torch.Size([6400, 2])
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # append a column tensor of h*w (i=0: size 6400,1), with values of stride (i=0: 8)
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    
    # anchor_points = [torch.Size([6400, 2]), torch.Size([1600, 2]), torch.Size([400, 2])]
    # stride tensor = [torch.Size([6400, 1]), torch.Size([1600, 1]), torch.Size([400, 1])] #column tensor for the respective strides of anchor 
    # stride=8: h,w=80  stride=16: h,w=40  stide=32: h,w=20 (i think) 
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)

# width, height -> x, y
def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300  # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs

def set_non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    batch_size, num_sets, num_features, num_proposals = prediction.shape  # New shape [8, 2, 5, 8400]
    outputs = [torch.zeros((0, 6), device=prediction.device) for _ in range(batch_size)]

    for batch_index in range(batch_size):
        batch_output = torch.zeros((0, 6), device=prediction.device)
        for set_index in range(num_sets):
            x = prediction[batch_index, set_index]  # Working with one set of predictions [5, 8400]
            x = x.permute(1, 0).contiguous().view(-1, 5)  # Reshape to [8400, 5]

            # Apply confidence threshold
            xc = x[:, 4] > conf_threshold

            if xc.any():
                x = x[xc]  # Filtered by confidence

                # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
                box = wh2xy(x[:, :4])

                # Prepare for batched NMS: [boxes, scores, class]
                scores = x[:, 4]
                cls = torch.zeros((len(scores), 1), device=prediction.device)  # Dummy class column
                nms_input = torch.cat((box, scores.unsqueeze(1), cls), dim=1)

                # Apply NMS
                keep = torchvision.ops.nms(nms_input[:, :4], nms_input[:, 4], iou_threshold)

                # Gather kept detections
                kept_detections = nms_input[keep]

                # Concatenate detections from both sets
                batch_output = torch.cat((batch_output, kept_detections), dim=0)

        # Limit the number of detections to max_det
        if len(batch_output) > 300:
            batch_output = batch_output[:300]

        outputs[batch_index] = batch_output

    return outputs

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class ComputeLoss:
    def __init__(self, model, params):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device
        self.params = params

        self.k = m.k # OPMP number of preds per anchor

        # task aligned assigner
        self.top_k = 10  # The number of top candidates to consider.
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)

    def __call__(self, outputs, targets):
        # x = outputs = [torch.Size([16, 65, 80, 80]), torch.Size([16, 65, 40, 40]), torch.Size([16, 65, 20, 20])]
        # OPMP: x = [torch.Size([16, 130, 80, 80]), torch.Size([16, 130, 40, 40]), torch.Size([16, 130, 20, 20])]
        x = outputs[1] if isinstance(outputs, tuple) else outputs

        # .shape = .size
        # x[0].shape[0]=16
        # self.no=outputs per anchor=65
        # output = cat([16, 65, 80*80], [16, 65, 40*40], [16, 65, 20*20]) = [16, 65, 8400]
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2) # OPMP: [16, 130, 8400]

        # split output into 4*16=64 and 1, using dimension 1 (65)
        pred_output, pred_scores = output.split((self.k * 4 * self.dfl_ch, self.nc * self.k), 1)
        # swap dim 1 and 2 -> pred_output=torch.Size([16, 8400, 64]), pred_scores=torch.Size([16, 8400, 1])
        pred_output = pred_output.permute(0, 2, 1).contiguous() # OPMP: [16, 8400, 128]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # OPMP: [16, 8400, 2]

        # size = tensor([80., 80.])
        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        # stride = tensor([ 8., 16., 32.])
        # size = tensor([640., 640.])
        size = size * self.stride[0]

        # anchor_points is [x,y] of a anchor point, not bbox
        # combined anchor points for all features and respective strides
        # 8400 points
        # anchor_points = [torch.Size([6400, 2]), torch.Size([1600, 2]), torch.Size([400, 2])]
        # stride tensor = [torch.Size([6400, 1]), torch.Size([1600, 1]), torch.Size([400, 1])] #column tensor for the respective strides of anchor 
        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5) # [8400, 2], [8400, 1]

        # targets
        if targets.shape[0] == 0:  # if no bbox
            gt = torch.zeros(pred_scores.shape[0], 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index of targets
            _, counts = i.unique(return_counts=True) # tensor of number of target bbox every image (index)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 5, device=self.device) # torch.Size([16, max count, 5])
            for j in range(pred_scores.shape[0]): # 16
                matches = i == j  # true false tensor of size targets
                n = matches.sum() # number of targets for image index j
                if n:
                    # groundtruth for image j = targets matching image id i.e. targets[1:] i.e. targets[class, x_center, y_center, width, height]
                    gt[j, :n] = targets[matches, 1:]
            
            # convert from [x_center, y_center, width, height] to [top_left_x, top_left_y, bot_right_x, bot_right_y]
            # multiplied to become a 640*640 image
            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]])) # size[[1, 0, 1, 0]]) = tensor([640., 640., 640., 640.])
        #  torch.Size([16, max count, 1]), torch.Size([16, max count, 4])
        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy # split gt into 5 chunks, 1 to cls 4 to xyxy, along dim2
        # mask_gt creates a filter for valid gt boxes, as previously the size of list set to counts.max to accommodate image with most targets
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) # .sum(dim=2, keepdim=True)

        # boxes
        b, a, c = pred_output.shape # pred_output=torch.Size([16, 8400, 128]) # batch, anchor, channels
        pred_bboxes = pred_output.view(b, a, 4 * self.k, c // 4 // self.k).softmax(3) # [16, 8400, 4, 16]
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype)) # [16, 8400, 4] # xyxy, (b, h*w, 4) # OPMP: [16, 8400, 8]

        a, b, c, d = torch.split(pred_bboxes, 2, -1) # torch.Size([16, 8400, 2]) torch.Size([16, 8400, 2])
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b, anchor_points - c, anchor_points + d), -1) # torch.Size([16, 8400, 4]) # scale it to image size

        scores = pred_scores.detach().sigmoid() # [16, 8400, 1] # OPMP: [16, 8400, 2] shape(bs, num_total_anchors, num_classes)
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype) # multiply pred box w stride [16, 8400, 4] -> [16, 8400, 4] # OPMP: [16, 8400, 8]


        '''
        target_bboxes: [16, 8400, 2, 4]
        target_scores: [16, 8400, 2, 1]
        fg_mask: [16, 8400, 2]
        '''
        # # [16, 8400, 4], [16, 8400, 1], [16, 8400]
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor) # TOOD
        '''
        targets already assigned using algin metrics with pred scores, may not need permutation with targets
        '''
        
        '''
        target_bboxes /= stride_tensor.unsqueeze(-1) -> [16, 8400, 2, 4]
        target_scores_sum0 = target_scores[:,:,0:1,:].sum().reshape(1)
        target_scores_sum1 = target_scores[:,:,1:2,:].sum().reshape(1)
        target_scores_sum = torch.cat(target_scores_sum0, target_scores_sum1) -> tensor(a, b)
        '''
        target_bboxes /= stride_tensor.unsqueeze(-1) # [16, 8400, 4])
        target_scores_sum0 = target_scores[:,:,0:1,:].sum().reshape(1)
        target_scores_sum1 = target_scores[:,:,1:2,:].sum().reshape(1)
        target_scores_sum = torch.cat((target_scores_sum0, target_scores_sum1), dim=0)
        # target_scores_sum = target_scores.sum() # tensor(722.0627)

        '''
        scrap permutations
        loss0: pred0 pred1, target
        loss1: pred1 pred0, target
        return min of sum(loss0s) or sum(loss1s)

        predscores: [16, 8400, 2] -> unsqueeze(-1) -> [16, 8400, 2, 1]
        targetscores: [16, 8400, 2, 1]
        loss_cls(bce): [16, 8400, 2, 1]
        loss_cls: tensor(a, b)
        '''
        # cls loss
        loss_cls = self.bce(pred_scores.unsqueeze(-1), target_scores.to(pred_scores.dtype)) # torch.Size([16, 8400, 1])
        loss_cls0 = loss_cls[:, :, 0:1, :].sum().reshape(1)
        loss_cls1 = loss_cls[:, :, 1:2, :].sum().reshape(1) #alw inf, maybe cuz of predscores shape
        loss_cls = torch.cat((loss_cls0, loss_cls1), dim=0) / target_scores_sum
        print('losscls', loss_cls)

        # loss_cls = loss_cls.sum() / target_scores_sum # tensor(7.2681)

        # box loss
        loss_box = torch.zeros(2, device=self.device) # initialize torch.Size([1])
        loss_dfl = torch.zeros(2, device=self.device) # initialize torch.Size([1])
        if fg_mask.sum():
            '''
            target_scores: [16, 8400, 2, 1]
            fg_mask: [16, 8400, 2]

            pred_bboxes: [16, 8400, 8]
            target_bboxes: [16, 8400, 2, 4]
            pred_output: [16, 8400, 128]
            '''

            target_scores0, target_scores1 = torch.split(target_scores, 1, -2) # [16, 8400, 1, 1]
            target_scores0 = target_scores0.squeeze(-2) # [16, 8400, 1]
            target_scores1 = target_scores1.squeeze(-2)

            fg_mask0, fg_mask1 = torch.split(fg_mask, 1, -1) # [16, 8400, 1]
            fg_mask0 = fg_mask0.squeeze() # [16, 8400]
            fg_mask1 = fg_mask1.squeeze()
            # IoU loss
            # masked_select: apply fg_mask, result in 1 dimension vector of size x

            weight0 = torch.masked_select(target_scores0.sum(-1), fg_mask0).unsqueeze(-1)
            weight1 = torch.masked_select(target_scores1.sum(-1), fg_mask1).unsqueeze(-1)
            print('weight', weight0.size())
            # weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1) # [x, 1]
            
            pred_bboxes0, pred_bboxes1 = torch.split(pred_bboxes, 4, -1) # [16, 8400, 4]
            target_bboxes0, target_bboxes1 = torch.split(target_bboxes, 1, -2)
            target_bboxes0 = target_bboxes0.squeeze(-2)
            target_bboxes1 = target_bboxes1.squeeze(-2)
            print('target bbox', target_bboxes1.size())

            loss_box0 = self.iou(pred_bboxes0[fg_mask0], target_bboxes0[fg_mask0])
            loss_box0 = ((1.0 - loss_box0) * weight0).sum() / target_scores_sum[0]
            loss_box1 = self.iou(pred_bboxes1[fg_mask1], target_bboxes1[fg_mask1])
            loss_box1 = ((1.0 - loss_box1) * weight1).sum() / target_scores_sum[1]
        
            # loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask]) # [x, 1]
            # loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum # tensor(0.6298)
            
            # DFL loss
            a0, b0 = torch.split(target_bboxes0, 2, -1)
            a1, b1 = torch.split(target_bboxes1, 2, -1) # [16, 8400, 2]

            target_lt_rb0 = torch.cat((anchor_points - a0, b0 - anchor_points), -1)
            target_lt_rb1 = torch.cat((anchor_points - a1, b1 - anchor_points), -1)
            # target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb0 = target_lt_rb0.clamp(0, self.dfl_ch - 1.01)
            target_lt_rb1 = target_lt_rb1.clamp(0, self.dfl_ch - 1.01)
            #target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            
            '''
            fg_mask: [16, 8400, 2]
            pred_output: [16, 8400, 128]
            '''
            pred_output0, pred_output1 = torch.split(pred_output, 64, -1)
            loss_dfl0 = self.df_loss(pred_output0[fg_mask0].view(-1, self.dfl_ch), target_lt_rb0[fg_mask0])
            loss_dfl1 = self.df_loss(pred_output1[fg_mask1].view(-1, self.dfl_ch), target_lt_rb1[fg_mask1])
            # loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            loss_dfl0 = (loss_dfl0 * weight0).sum() / target_scores_sum[0]
            loss_dfl1 = (loss_dfl1 * weight1).sum() / target_scores_sum[1] 
            # loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        loss_cls0 *= self.params['cls']
        loss_cls1 *= self.params['cls']
        loss_box0 *= self.params['box']
        loss_box1 *= self.params['box']
        loss_dfl0 *= self.params['dfl']
        loss_dfl1 *= self.params['dfl']

        loss0 = loss_cls0 + loss_box0 + loss_dfl0
        loss1 = loss_cls1 + loss_box1 + loss_dfl1
        
        print(min(loss0, loss1))
        return min(loss0, loss1)  # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        """
        Task-aligned One-stage Object Detection assigner
        
        assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
        classification and localization information.

        pred_scores: [16, 8400, 1] -> [16, 8400, 2]
        pred_bboxes: [16, 8400, 4] -> [16, 8400, 8]
        true_labels: [16, max count, 1]
        true_bboxes: [16, max count, 4]
        true_mask: [16, max count, 1] (true for indices where there is a valid gt)
        anchors: [8400, 2]
        """
        self.bs = pred_scores.size(0) # 16
        self.num_max_boxes = true_bboxes.size(1) # 64

        print('true_mask', true_mask.size())
        print('anchors', anchors.size())

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))

        #i:  [torch.Size([16, 64]), ([16, 64])]
        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)

        '''
        target overlaps: [16,64,8400,2] (ious for 2 preds)
        '''
        overlap0 = self.iou(true_bboxes.unsqueeze(2), pred_bboxes[...,0:4].unsqueeze(1))
        overlap1 = self.iou(true_bboxes.unsqueeze(2), pred_bboxes[...,4:8].unsqueeze(1))
        overlaps = torch.cat((overlap0, overlap1), dim=-1)
        # overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1)) # [16, 64, 8400, 1] 
        overlaps = overlaps.squeeze(3).clamp(0) # [16, 64, 8400])
        # pred_scores[i[0], :, i[1]] = bbox scores = Get the scores of each grid for each gt cls # size = [16, maxtargets, 8400]
        '''
        target align_metrics: [16,64,8400,2] (metric for 2 pred scores)
        '''
        align_metric0 = pred_scores[...,0:1][i[0], :, i[1]].pow(self.alpha) * overlaps[...,0:1].squeeze(-1).pow(self.beta)
        align_metric1 = pred_scores[...,1:2][i[0], :, i[1]].pow(self.alpha) * overlaps[...,1:2].squeeze(-1).pow(self.beta)
        align_metric = torch.cat((align_metric0.unsqueeze(-1), align_metric1.unsqueeze(-1)), dim=-1)
        # align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta) # [16, 64, 8400]
        
        bs, n_boxes, _ = true_bboxes.shape # 16, 64
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom # ([1024, 1, 2]) ([1024, 1, 2])
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2) # [1024, 8400, 4]
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9) # [16, 64, 8400]

        '''
        target mask_in_gts: [16, 64, 8400, 2] (repeat)
        target metrics: [16, 64, 8400, 2]

        target top_k_metrics, top_k_indices: [16, 64, 10, 2] (maybe do for metrics[...:k,k+1] to get indiv topk, then cat)
        '''
        mask_in_gts = mask_in_gts.unsqueeze(-1).repeat(1,1,1,2) 
        metrics = align_metric * mask_in_gts # [16, 64, 8400]
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool() # [16, 64, 10]
        num_anchors = anchors.shape[0] # 8400

        top_k_metrics0, top_k_indices0 = torch.topk(metrics[...,0:1].squeeze(-1), self.top_k, dim=-1, largest=True)
        top_k_metrics1, top_k_indices1 = torch.topk(metrics[...,1:2].squeeze(-1), self.top_k, dim=-1, largest=True)
        top_k_metrics = torch.cat((top_k_metrics0.unsqueeze(-1), top_k_metrics1.unsqueeze(-1)), dim=-1)
        top_k_indices = torch.cat((top_k_indices0.unsqueeze(-1), top_k_indices1.unsqueeze(-1)), dim=-1)
        # top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True) # [16, 64, 10] [16, 64, 10]
        if top_k_mask is None:
            print('top_k_mask is None')
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])

        '''
        continued from above section (talking abt top k METRICS)
        target top_k_mask: [16, 64, 10, 2] (repeat top_k_mask)
        target top_k_indices: [16, 64, 10, 2]
        one_hot_topk: one_hot(top_k_indices, num_anchors) = [16, 64, 10, 8400] -> at 2 one_hots -> target: [16, 2, 64, 10, 8400]
        target is_in_top_k: [16, 2, 64, 8400] (one_hot_topk.sum(-2))
        '''
        top_k_mask = top_k_mask.unsqueeze(-1).repeat(1,1,1,2)
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0) # [16, 64, 10]
        is_in_top_k0 = one_hot(top_k_indices[...,0:1].squeeze(), num_anchors) # [16, 64, 8400]
        is_in_top_k1 = one_hot(top_k_indices[...,1:2].squeeze(), num_anchors)
        is_in_top_k = torch.cat((is_in_top_k0.unsqueeze(1), is_in_top_k1.unsqueeze(1)), dim=1)
        is_in_top_k = is_in_top_k.sum(-2)
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k) # [16, 64, 8400]
        mask_top_k = is_in_top_k.to(metrics.dtype) # [16, 64, 8400]
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        '''
        target mask_top_k = [16, 2, 64, 8400].permute(0,2,3,1) = [16, 64, 8400, 2]
        split and flatten mask_top_k and mask_in_gts to 0 and 1: [16, 64, 8400, 2] -> [16, 64, 8400, 1] -> [16, 64, 8400]
        mask_pos 0, 1 = [16, 64, 8400]*[16, 64, 8400]*[16, 64, 1] = [16, 64, 8400]
        mask_pos = unsqueeze(-1), cat 0,1 -> [16, 64, 8400, 2]
        target mask_pos: [16, 64, 8400, 2]
        '''
        mask_top_k = mask_top_k.permute(0,2,3,1)
        mask_pos = mask_top_k * mask_in_gts * true_mask.unsqueeze(-1) # [16, 64, 8400]*[16, 64, 8400]*[16, 64, 1] = [16, 64, 8400]
        '''
        target fg_mask = mask_pos.sum(-3)
        target fg_mask = [16, 8400, 2]
        '''
        fg_mask = mask_pos.sum(-3) # [16, 8400]
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes # non-opmp will select the gt for highest iou
            '''
            fg_mask split 0,1 = [16, 8400, 1] [16, 8400, 1]
            fg_mask 0,1 permute = [16, 1, 8400] [16, 1, 8400]

            mask_multi_gts 0,1 = [16, 99, 8400] [16, 99, 8400] -> cat -> [16, 99, 2, 8400] -> permute [16, 99, 8400, 2]
            overlaps: [16, 99, 8400, 2]
            max_overlaps_idx : [16, 8400, 2]
            is_max_overlaps: [16, 8400, 2, 99] -> permute -> # [16, 99, 8400, 2]
            mask_pos: [16, 99, 8400, 2]
            fg_mask: mask_pos.sum(-3) = [16, 8400, 2]
            '''
            fg_mask0, fg_mask1 = fg_mask.split(1, dim=-1)
            fg_mask0 = fg_mask0.permute(0,2,1)
            fg_mask1 = fg_mask1.permute(0,2,1)
            
            mask_multi_gts0 = (fg_mask0 > 1).repeat([1, self.num_max_boxes, 1]).unsqueeze(-2)
            mask_multi_gts1 = (fg_mask1 > 1).repeat([1, self.num_max_boxes, 1]).unsqueeze(-2)
            mask_multi_gts = torch.cat((mask_multi_gts0, mask_multi_gts1), dim=-2).permute(0,1,3,2)
            # mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1]) # [16, 99, 8400]
            max_overlaps_idx = overlaps.argmax(1) # [16, 8400] # for each image, each anchor uses which target(idx)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes) # [16, 8400, 99]
            is_max_overlaps = is_max_overlaps.permute(0, 3, 1, 2).to(overlaps.dtype) # [16, 99, 8400]
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) # [16, 99, 8400]
            fg_mask = mask_pos.sum(-3) # [16, 8400]
        
        '''
        target_gt_idx: mask_pos.argmax(-3) -> [16, 8400, 2]
        '''
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-3)  # (b, h*w) # [16, 8400]
        '''
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[:,None, None] (16,1,1)
        target_gt_idx: [16, 8400, 2]

        target_labels = true_labels.long().flatten()[target_gt_idx.reshape(-1, 2)]
        target_labels = target_labels.view(target_gt_idx.size())
        target target_labels: [16, 8400, 2]
        '''
        # assigned target labels, (b, 1) 
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[:,None, None]  # [16, 1]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes # [16, 8400]
        target_labels = true_labels.long().flatten()[target_gt_idx.reshape(-1, 2)] # [16, 8400]
        target_labels = target_labels.view(target_gt_idx.size())

        '''
        target target_bboxes: [16, 8400, 2, 4]
        '''
        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx] # [16, 8400, 4]
        '''
        target_labels.clamp(0):  [16, 8400, 2]
        target_scores: [16, 8400, 2, 1]

        fg_mask: [16, 8400, 2]
        fg_scores_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 1, self.nc) -> [16, 8400, 2, 1]
        '''        
        # assigned target scores
        target_labels.clamp(0) # [16, 8400]
        target_scores = one_hot(target_labels, self.nc) # [16, 8400, 1]
        fg_scores_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 1, self.nc) # [16, 8400, 1]
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0) # [16, 8400, 1]

        '''
        align metric: [16,99,8400,2]
        mask_pos: [16, 99, 8400, 2]
        pos_align_metrics = align_metric.amax(axis=-2, keepdim=True) -> [16, 99, 1, 2]
        pos_overlaps = (overlaps * mask_pos).amax(axis=-2, keepdim=True) -> [16, 99, 1, 2]
        norm_align_metric = ...amax(-3) -> [16, 8400, 2]
        target_scores: [16, 8400, 2, 1]
        '''
        # normalize
        align_metric *= mask_pos # [16, 64, 8400]
        pos_align_metrics = align_metric.amax(axis=-2, keepdim=True) # [16, 64, 1]
        pos_overlaps = (overlaps * mask_pos).amax(axis=-2, keepdim=True) # [16, 64, 1])
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-3) # [16, 8400]
        norm_align_metric = norm_align_metric.unsqueeze(-1) # [16, 8400, 1]
        target_scores = target_scores * norm_align_metric # [16, 8400, 1]
        '''
        target_bboxes: [16, 8400, 2, 4]
        target_scores: [16, 8400, 2, 1]
        fg_mask: [16, 8400, 2]
        '''
        return target_bboxes, target_scores, fg_mask.bool()
    '''
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        """
        Task-aligned One-stage Object Detection assigner
        
        assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
        classification and localization information.

        pred_scores: [16, 8400, 2]
        pred_bboxes: [16, 8400, 8]
        true labels: [16, max count, 1]
        true_bboxes: [16, max count, 4]

        """
        self.bs = pred_scores.size(0) # 16 # batch size
        self.num_max_boxes = true_bboxes.size(1) # 64 # max targets, from one image i.e. counts.max i think
        self.k = 2

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))

        #i:  [([16, 64]), ([16, 64])]
        #i:  [16, 2, 107], [16, 2, 107] OPMP, diff batch
        i = torch.zeros([2, self.bs, self.k,  self.num_max_boxes], dtype=torch.long) # [2, 16, 2, 107]
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes).unsqueeze(1).repeat(1,self.k,1) # [16, 2, 107]
        i[1] = true_labels.long().squeeze(-1).unsqueeze(1).repeat(1,2,1)

        # need to allocate up to k gt boxeswhen iou>threshold
        # split pred_bboxes into list of bboxes
        # indiv_pred_bboxes = list(torch.split(pred_bboxes, 4, -1)) # [[16, 8400, 4],[16, 8400, 4]]
        # overlaps = [] # [[16, 89, 8400],[16, 89, 8400]]
        # for i in range(self.k):
        #     overlap = self.iou(true_bboxes.unsqueeze(2), indiv_pred_bboxes[i].unsqueeze(1))
        #     overlaps.append(overlap.squeeze(3).clamp(0))

        overlaps = torch.zeros([self.bs, self.num_max_boxes, pred_scores.size(1), self.k]).to(true_bboxes.device) #[16, maxtargets, 8400, 2]
        align_metric = torch.zeros(self.bs, self.k*self.num_max_boxes, pred_scores.size(1)).to(true_bboxes.device) #[16, 2*maxtargets, 8400]
        for k in range(self.k):
            overlaps[..., k:k+1] = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.split(4,-1)[k].unsqueeze(1)) # [16, maxtargets, 8400, 2]
            print('pred_scores[..., k:k+1]', pred_scores[..., k:k+1].size()) # [16, 8400, 1]
            print('pred_scores[..., k:k+1][i[0], :, i[1]]', pred_scores[..., k:k+1][i[0], :, i[1]].size()) 
            print('overlaps[..., k:k+1].squeeze()',overlaps[..., k:k+1].squeeze().size()) 
            a = pred_scores[..., k:k+1][i[0], :, i[1]].pow(self.alpha) # [16, 2, 107, 8400]
            b = overlaps[..., k:k+1].squeeze().pow(self.beta) # [16, 107, 8400]
            align_metric[16, k*self.num_max_boxes:k*self.num_max_boxes+self.num_max_boxes, pred_scores.size(1)] = a * b # [16, 64, 8400]
        # overlaps = overlaps.squeeze(3).clamp(0) # [16, maxtargets, 8400])
        print('iou', overlaps.size())
        print('align metric', align_metric.size())
        # pred scores: [16, 8400, 2]
        bs, n_boxes, _ = true_bboxes.shape # 16, 64
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom # ([1024, 1, 2]) ([1024, 1, 2])
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2) # [1024, 8400, 4]
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9) # [16, 64, 8400]
        metrics = align_metric * mask_in_gts # [16, 64, 8400]
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool() # [16, 64, 10]
        num_anchors = metrics.shape[-1] # 8400
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True) # [16, 64, 10] [16, 64, 10]
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0) # [16, 64, 10]
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2) # [16, 64, 8400]
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k) # [16, 64, 8400]
        mask_top_k = is_in_top_k.to(metrics.dtype) # [16, 64, 8400]
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask # [16, 64, 8400]
        mask_pos = mask_pos.unsqueeze(1).repeat(1,self.k,1,1) # [16, k, maxtargets, 8400]

        fg_mask = mask_pos.sum(-2) # [16, 8400]
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes # opmp should select k gts with iou > threshold
            print('more than 1 gt assigned')
            print('fg masks', fg_mask)
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1]) # [16, 64, 8400]
            print('mask multi gts', mask_multi_gts.size())
            mask_multi_gts = mask_multi_gts.unsqueeze(1).repeat(1,self.k,1,1)# [16, k, maxtargets, 8400]
            print('new mask multi gts', mask_multi_gts.size())
            # for each image, each anchor uses which target(idx) # [16, 8400]
            _, max_overlaps_idx = torch.topk(overlaps, self.k, -2) # [16, 2, 8400] # for each image, each anchor has 2 gts
            max_overlaps_idx = max_overlaps_idx.permute(0,2,1) # [16, 8400, 2]
            # max_overlaps_idx = overlaps.argmax(1) # [16, 8400] # should use topk instead of argmax
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes) # [16, 8400, 64] top k=2 max iou: # [16, 8400, 2, maxtargets]
            is_max_overlaps = is_max_overlaps.permute(0, 3, 2, 1).to(overlaps.dtype) # [16, maxtargets, 2 , 8400]
            print('ismaxovalaps', is_max_overlaps.size())
            # is_max_overlaps = is_max_overlaps.reshape(self.bs, -1, is_max_overlaps.shape[-1]) # [16, 2*maxtargets, 8400]
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) # [16, 2*maxtargets, 8400]
            fg_mask = mask_pos.sum(-2) # [16, 8400]
            print('fg mask', fg_mask.size())

        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w) # [16, 8400]
        print('target gt index', target_gt_idx)
        # assigned target labels, (b, 1) 
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]  # [16, 1]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes # [16, 8400]
        target_labels = true_labels.long().flatten()[target_gt_idx] # [16, 8400]
        print('batch index', batch_index.size())
        print('target gt idx', target_gt_idx.size())
        print('target labels', target_labels.size())

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx] # [16, 8400, 4]
        print('target bboxes', target_bboxes.size())

        # assigned target scores
        target_labels.clamp(0) # [16, 8400]
        target_scores = one_hot(target_labels, self.nc) # [16, 8400, 1]
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc) # [16, 8400, 1]
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0) # [16, 8400, 1]
        print('fg_scores_mask', fg_scores_mask.size())
        print('target_scoress', target_scores.size())

        # normalize
        align_metric *= mask_pos # [16, 64, 8400]
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True) # [16, 64, 1]
        pos_overlaps = (overlaps.repeat(1, self.k, 1) * mask_pos).amax(axis=-1, keepdim=True) # [16, 64, 1])
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2) # [16, 8400]
        norm_align_metric = norm_align_metric.unsqueeze(-1) # [16, 8400, 1]
        target_scores = target_scores * norm_align_metric # [16, 8400, 1]

        # # [16, 8400, 4], [16, 8400, 1], # [16, 8400]
        # OPMP target: [16, 8400, 8], [16, 8400, 2], # [16, 8400]
        return target_bboxes, target_scores, fg_mask.bool()
    '''
    

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1, box2, eps=1e-7): # DIoU + aspect ratio
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
