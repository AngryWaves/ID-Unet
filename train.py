# torch and visulization
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader


# metric, loss .etc
from Ablation import ID_UNet_ADD
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.parse_args_train import  parse_args
from ID_UNet import ID_UNet


class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args = args
        self.ROC = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir
        self.device = torch.device(args.gpus if torch.cuda.is_available() else "cpu")

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)
            self.val_img_ids = val_img_ids

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=args.base_size,
                                         crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir, img_id=val_img_ids, base_size=args.base_size,
                                         crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size,
                                     shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size,
                                     num_workers=args.workers, drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        model = ID_UNet()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model.to(self.device)

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Evaluation metrics
        self.best_IoU, self.best_nIoU, self.best_PD, self.best_FA = 0, 0, 0, 1
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.train_loss, self.test_loss = 0, 0
        self.prediction_rate, self.recall_rate, self.F1 = 0, 0, 0

    # Training
    def training(self, epoch, writer):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.to(self.device)
            labels = labels.to(self.device)
            if args.deep_supervision == 'True':
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
            else:
               pred = self.model(data)
               loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg
        writer.add_scalar('train_loss', losses.avg, epoch)

    # Testing
    def testing(self, epoch, writer):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA = PD_FA(1, args.ROC_thr)
        losses = AverageMeter()
        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.to(self.device)
                labels = labels.to(self.device)
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                self.PD_FA.update(pred, labels)
                pixF1, pixPrecision, pixRecall, mean_IOU, nIOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
            test_loss = losses.avg

        pixF1, pixPrecision, pixRecall, mean_IOU, nIOU = self.mIoU.get()
        FA, PD = self.PD_FA.get(len(self.val_img_ids))
        ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
        IoU, nIoU = mean_IOU, nIOU

        # tensorboard
        writer.add_scalar('test_loss', losses.avg, epoch)
        writer.add_scalar('IOU', IoU, epoch)
        writer.add_scalar('nIou', nIoU, epoch)
        writer.add_scalar('PD', PD[0], epoch)
        writer.add_scalar('FA', FA[0], epoch)
        writer.add_scalar('pixF1', pixF1, epoch)
        writer.add_scalar('pixPrecision', pixPrecision, epoch)
        writer.add_scalar('pixRecall', pixRecall, epoch)

        best = "IoU:%.4f, nIoU:%.4f, PD:%.4f, FA:%.4g, epoch:%d" \
               ", pixF1:%.4f, pixPrecision:%.4f, pixRecall:%.4f" \
               % (IoU, nIoU, PD[0], FA[0], epoch, pixF1, pixPrecision, pixRecall)
        if self.best_IoU < IoU:
            self.best_IoU = IoU
            # save high-performance model
            save_model(IoU, self.best_iou, self.save_dir, self.save_prefix,
                       self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
            writer.add_text('1.best_IoU', best, 0)
        if self.best_nIoU < nIoU:
            self.best_nIoU = nIoU
            writer.add_text('4.best_nIoU', best, 0)
        if self.best_PD < PD[0]:
            self.best_PD = PD[0]
            writer.add_text('5.best_PD', best, 0)
        if self.best_FA > FA[0]:
            self.best_FA = FA[0]
            writer.add_text('6.best_FA', best, 0)


def main(args):
    trainer = Trainer(args)
    import time
    # writer = SummaryWriter(os.path.join("logs", args.dataset, '_'.join(time.ctime().replace(":", "_").split(" "))))
    time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    writer = SummaryWriter(os.path.join("logs", args.dataset, time))
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch, writer)
        trainer.testing(epoch, writer)


if __name__ == "__main__":
    args = parse_args()
    main(args)





