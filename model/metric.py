import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy

base_size = 256

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):
        tp_rates    = self.tp_arr / (self.pos_arr + np.spacing(1))
        fp_rates    = self.fp_arr / (self.neg_arr + np.spacing(1))

        recall      = self.tp_arr / (self.pos_arr + np.spacing(1))
        precision   = self.tp_arr / (self.class_pos + np.spacing(1))

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

# class ROCMetric():
#     """Computes pixAcc and mIoU metric scores
#     """
#     def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
#         super(ROCMetric, self).__init__()
#         self.nclass = nclass
#         self.bins = bins
#         self.tp_arr = np.zeros(self.bins+1)
#         self.t_arr = np.zeros(self.bins+1)
#         self.p_arr = np.zeros(self.bins+1)
#         # self.fpr = np.zeros(self.bins+1)
#         # self.tpr = np.zeros(self.bins+1)
#         # self.reset()
#
#     def update(self, preds, labels):
#         for iBin in range(self.bins+1):
#             # score_thresh = (iBin + 0.0) / self.bins
#             score_thresh = iBin*0.1
#             # print(iBin, "-th, score_thresh: ", score_thresh)
#             tp, t, p = Metric(preds, labels, self.nclass, score_thresh)
#             self.tp_arr[iBin] += tp
#             self.t_arr[iBin] += t
#             self.p_arr[iBin] += p
#
#         # fpr, tpr, thresholds = metrics.roc_curve(preds.ravel(), labels.ravel())
#         # self.fpr += fpr
#         # self.tpr += tpr
#
#
#     def get(self):
#         precision = self.tp_arr / (self.t_arr + np.spacing(1))
#         recall = self.tp_arr / (self.p_arr + np.spacing(1))
#         return precision, recall
#         # return self.fpr, self.tpr
#
#     def reset(self):
#         self.tp_arr = np.zeros(self.bins + 1)
#         self.t_arr = np.zeros(self.bins + 1)
#         self.p_arr = np.zeros(self.bins + 1)

class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
    def update(self, predsss, labelsss):

        for i in range(predsss.size(0)):
            preds = predsss[i]
            labels = labelsss[i]

            for iBin in range(self.bins+1):
                score_thresh = iBin * (base_size/self.bins)
                predits  = np.array((preds > score_thresh).cpu()).astype('int64')
                predits  = np.reshape (predits,  (base_size,base_size))
                labelss = np.array((labels).cpu()).astype('int64') # P
                labelss = np.reshape (labelss , (base_size,base_size))

                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(labelss , connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[iBin]    += len(coord_label)
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match   = []
                self.dismatch         = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            self.distance_match.append(distance)
                            self.image_area_match.append(area_image)

                            del coord_image[m]
                            break

                self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                self.FA[iBin]+=np.sum(self.dismatch)
                self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((base_size * base_size) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled, positive = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_positive += positive
        self.total_inter += inter
        self.total_union += union
        for (pred, label) in zip(preds, labels):
            pred = torch.unsqueeze(pred, dim=0)
            label = torch.unsqueeze(label, dim=0)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            self.total_inter_n = np.append(self.total_inter_n, inter)
            self.total_union_n = np.append(self.total_union_n, union)


    def get(self):
        pixPrecision = 1.0 * self.total_correct / (np.spacing(1) + self.total_positive)
        pixPrecision = pixPrecision.mean()
        pixRecall = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        pixRecall = pixRecall.mean()
        F1 = 2 * pixPrecision * pixRecall / (np.spacing(1) + pixPrecision + pixRecall)
        mIoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = mIoU.mean()
        nIou = 1.0 * self.total_inter_n / (np.spacing(1) + self.total_union_n)
        nIou = nIou.mean()
        return F1, pixPrecision, pixRecall, mIoU, nIou

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_positive = 0
        self.total_inter_n = np.array([])
        self.total_union_n = np.array([])


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def Metric(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    t = (predict > 0).float().sum()
    p = (target > 0).float().sum()

    intersection = target * ((predict == target).float())
    tp = intersection.sum()

    # fp = (predict * ((predict != target).float())).sum()
    # tn = ((1 - predict) * ((predict == target).float())).sum()
    # fn = (((predict != target).float()) * (1 - predict)).sum()
    # pos = tp + fn
    # neg = fp + tn
    # class_pos= tp+fp
    return tp, t, p

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled, predict.sum()


def batch_intersection_union(output, target):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class Metrics:

    def __init__(self, bins=10):
        super(Metrics, self).__init__()
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        for bin in range(self.bins):
            correct, labeled, positive = batch_pix_accuracy(preds, labels)
            inter, union = batch_intersection_union(preds, labels, bin)
            self.total_correct += correct
            self.total_label += labeled
            self.total_positive += positive
            self.total_inter += inter
            self.total_union += union
            for (pred, label) in zip(preds, labels):
                pred = torch.unsqueeze(pred, dim=0)
                label = torch.unsqueeze(label, dim=0)
                inter, union = batch_intersection_union(pred, label, bin)
                self.total_inter_n = np.append(self.total_inter_n, inter)
                self.total_union_n = np.append(self.total_union_n, union)

    def get(self):
        pixPrecision = 1.0 * self.total_correct / (np.spacing(1) + self.total_positive)
        pixPrecision = pixPrecision.mean()
        pixRecall = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        pixRecall = pixRecall.mean()
        F1 = 2 * pixPrecision * pixRecall / (np.spacing(1) + pixPrecision + pixRecall)
        mIoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = mIoU.mean()
        nIou = 1.0 * self.total_inter_n / (np.spacing(1) + self.total_union_n)
        nIou = nIou.mean()
        return F1, pixPrecision, pixRecall, mIoU, nIou

    def reset(self):
        self.total_inter = np.zeros([self.bins+1])
        self.total_union = np.zeros([self.bins+1])
        self.total_correct = np.zeros([self.bins+1])
        self.total_label = np.zeros([self.bins+1])
        self.total_positive = np.zeros([self.bins+1])
        self.total_inter_n = np.array([])
        self.total_union_n = np.array([])