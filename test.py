# Basic module
from tqdm             import tqdm

from model.parse_args_test import  parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.parse_args_test import  parse_args
from ID_UNet import ID_UNet

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        model = ID_UNet()
        model = model.cuda()
        self.model = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        # checkpoint        = torch.load('result/' + args.model_dir)
        # checkpoint = torch.load('./test/IRSTD-1K_DNANet_25_02_2023_15_16_47_wDS/mIoU_DNANet_IRSTD-1K_epoch.pth.tar'
        #                                , map_location='cuda:0')
        # checkpoint = torch.load('./test/NUDT-SIRST_DNANet_13_02_2023_21_39_49_wDS/mIoU_DNANet_NUDT-SIRST_epoch.pth.tar'
        #                         , map_location='cuda:0')

        # checkpoint = torch.load('./test/IRSTD-1K_DNANet_25_02_2023_00_02_56_wDS/mIoU_DNANet_IRSTD-1K_epoch.pth.tar'
        #                         , map_location='cuda:0')
        # # checkpoint = torch.load('./test/NUDT-SIRST_DNANet_22_02_2023_10_23_48_wDS/mIoU_DNANet_NUDT-SIRST_epoch.pth.tar'
        # #                         , map_location='cuda:0')
        # self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                pixF1, pixPrecision, pixRecall, mean_IOU, nIOU = self.mIoU.get()
            pixF1, pixPrecision, pixRecall, mean_IOU, nIOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))

            IoU, nIoU = mean_IOU, nIOU
            print('test_loss: %.4f' % losses.avg)
            print('IoU: %.4f' % IoU)
            print('nIoU: %.4f' % nIoU)
            print('PD: %.4f' % PD[0])
            print('FA: %.4f' % FA[0])
            print('pixF1: %.4f' % pixF1)
            print('pixPrecision: %.4f' % pixPrecision)
            print('pixRecall: %.4f' % pixRecall)

            # scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})
            # print(acc, mean_IOU, nIOU, FA, PD, recall, precision)
            # print(ture_positive_rate)
            # print(false_positive_rate)
            # print(acc, mean_IOU, FA, PD, recall, precision)
            # save_result_for_test(dataset_dir, args.st_model, args.epochs, mean_IOU, recall, precision)

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





