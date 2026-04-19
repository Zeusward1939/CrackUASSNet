import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1001
os.environ["PYTHONHASHSEED"] = str(seed)
from datetime import datetime
from distutils.dir_util import copy_tree
import numpy as np
import torch
import argparse
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from utilities.dataloader.dataloaders import* 
from utilities.metrics import*
from main.utilities.losses_a import*
from main.utilities.losses_b import*
from utilities.pytorch_losses import dice_loss
from utilities.ramps import sigmoid_rampup
from UASS_model import model
from utilities.utilities import get_logger

 
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='U-Net_UASS')
parser.add_argument('--num_classes', type=int,  default=2)
parser.add_argument('--base_lr', type=float,  default=0.001, help='network learning rate')
parser.add_argument('--seed', type=int,  default=1001)
parser.add_argument('--ema_decay', type=float,  default=0.99)
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency_a', type=float, default=0.1)
parser.add_argument('--consistency_b', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
epochs = 800

num_classes = args.num_classes
no = args.no
kl_distance = nn.KLDivLoss(reduction='none')
log_sm = torch.nn.LogSoftmax(dim = 1)
ce_loss = CrossEntropyLoss()
base_lr = args.base_lr
iter_per_epoch = 25 
def get_current_consistency_weight_a(epoch):
    return args.consistency_a * sigmoid_rampup(epoch, args.consistency_rampup)
def get_current_consistency_weight_b(epoch):
    return args.consistency_b * sigmoid_rampup(epoch, args.consistency_rampup)

dataset_name = 'dataset_name'

class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.model = model
        self._init_logger()

    def _init_logger(self):
        
        log_dir = 'your/checkpoints/save/path'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
         
        self.model.to(device)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.00000001, patience=50, verbose=True)
              
        self.logger.info(
            "train_loader {} unlabeled_loader {} val_loader {} test_loader {} ".format(len(train_loader),
                                                                       len(unlabeled_loader),
                                                                       len(val_loader),
                                                                       len(test_loader)))
        print("Training process started!")
        print("============================================")

        iter_num = 0
        for epoch in range(1, epochs):

            running_train_ce_loss = 0.0
            running_train_dice_loss = 0.0
            running_train_loss = 0.0
            running_train_iou = 0.0
            running_train_dice = 0.0
            running_train_pl_loss = 0.0
            running_ep1_loss = 0.0
            running_ep2_loss = 0.0
            running_ep3_loss = 0.0
            running_ep4_loss = 0.0        
            running_uncertainity_loss = 0.0
            running_val_ce_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_loss = 0.0                               
            running_val_iou_1 = 0.0
            running_val_dice_1 = 0.0
            running_val_accuracy_1 = 0.0
                        
            optimizer_1.zero_grad()
            
            self.model.train()

            semi_dataloader = iter(zip(cycle(train_loader), cycle(unlabeled_loader))) 
                                 
            for iteration in range (1, iter_per_epoch):
                data = next(semi_dataloader)
 
                (inputs_S1, labels_S1), (inputs_U, labels_U) = data 

                if torch.isnan(inputs_S1).any() or torch.isnan(labels_S1).any():
                    print("Invalid values in inputs/labels!")
                    continue

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)

                inputs_U, labels_U = Variable(inputs_U), Variable(labels_U)
                inputs_U, labels_U = inputs_U.to(device), labels_U.to(device)

                self.model.train()

                outputs_ep1, outputs_ep2, outputs_ep3, outputs_ep4 = self.model(inputs_S1)
                outputs_ep1_soft = torch.softmax(outputs_ep1, dim=1)
                outputs_ep1_soft = torch.softmax(outputs_ep2, dim=1)
                outputs_ep2_soft = torch.softmax(outputs_ep3, dim=1)
                outputs_ep3_soft = torch.softmax(outputs_ep4, dim=1)

                un_outputs_ep1, un_outputs_ep2, un_outputs_ep3, un_outputs_ep4 = self.model(inputs_U)
                un_outputs_ep1_soft = torch.softmax(un_outputs_ep1, dim=1)
                un_outputs_ep2_soft = torch.softmax(un_outputs_ep2, dim=1)
                un_outputs_ep3_soft = torch.softmax(un_outputs_ep3, dim=1)
                un_outputs_ep4_soft = torch.softmax(un_outputs_ep4, dim=1)
                                
                loss_ce_ep1 = ce_loss(outputs_ep1, labels_S1.long())
                if torch.isnan(loss_ce_ep1):
                        print("NaN in CE loss!")
                        break
                loss_ce_ep2 = ce_loss(outputs_ep2, labels_S1.long())
                loss_ce_ep3 = ce_loss(outputs_ep3, labels_S1.long())
                loss_ce_ep4 = ce_loss(outputs_ep4, labels_S1.long())
                
                loss_dice_ep1 = dice_loss(labels_S1.unsqueeze(1), outputs_ep1)
                loss_dice_ep2 = dice_loss(labels_S1.unsqueeze(1), outputs_ep2)
                loss_dice_ep3 = dice_loss(labels_S1.unsqueeze(1), outputs_ep3)
                loss_dice_ep4 = dice_loss(labels_S1.unsqueeze(1), outputs_ep4)

                loss_ep1 = 0.5*(loss_ce_ep1 + loss_dice_ep1)
                loss_ep2 = 0.5*(loss_ce_ep2 + loss_dice_ep2)
                loss_ep3 = 0.5*(loss_ce_ep3 + loss_dice_ep3)
                loss_ep4 = 0.5*(loss_ce_ep4 + loss_dice_ep4)
                
                total_loss_ce = (loss_ce_ep1 + loss_ce_ep2 + loss_ce_ep3 + loss_ce_ep4)/4
                total_loss_dice = (loss_dice_ep1 + loss_dice_ep2 + loss_dice_ep3 + loss_dice_ep4)/4
                supervised_loss = (loss_ep1 + loss_ep2 + loss_ep3 + loss_ep4)/4
                
                preds = (un_outputs_ep1_soft + un_outputs_ep2_soft + un_outputs_ep3_soft+ un_outputs_ep4_soft)/4 # Averaging predictions on the unlabeled samples
                
                variance_ep1 = torch.sum(kl_distance(log_sm(un_outputs_ep1), preds), dim=1) # Uncertainty map between average and each prediction
                exp_variance_ep1 = torch.exp(-variance_ep1)

                variance_ep2 = torch.sum(kl_distance(log_sm(un_outputs_ep2), preds), dim=1)
                exp_variance_ep2 = torch.exp(-variance_ep2)

                variance_ep3 = torch.sum(kl_distance(log_sm(un_outputs_ep3), preds), dim=1)
                exp_variance_ep3 = torch.exp(-variance_ep3)

                variance_ep4 = torch.sum(kl_distance(log_sm(un_outputs_ep4), preds), dim=1)
                exp_variance_ep4 = torch.exp(-variance_ep4)

                ave_var = (variance_ep1 + variance_ep2 + variance_ep3 + variance_ep4) /4  

                l_uncert = torch.mean(ave_var) 

                lbl_weight = np.random.dirichlet(np.ones(4),size=1)[0]
                un_lbl_pseudo = torch.argmax((lbl_weight[0]*un_outputs_ep1_soft.detach() + \
                                              lbl_weight[1]*un_outputs_ep2_soft.detach() + \
                                              lbl_weight[2]*un_outputs_ep3_soft.detach() + \
                                              lbl_weight[3]*un_outputs_ep4_soft.detach()), 
                                              dim=1, keepdim=False)

                pl_1 = 0.5*(ce_loss(un_outputs_ep1, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_ep1))
                pl_2 = 0.5*(ce_loss(un_outputs_ep2, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_ep2)) 
                pl_3 = 0.5*(ce_loss(un_outputs_ep3, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_ep3))
                pl_4 = 0.5*(ce_loss(un_outputs_ep4, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_ep4))
                
                pl_1_loss = torch.mean(pl_1 * exp_variance_ep1) 
                pl_2_loss = torch.mean(pl_2 * exp_variance_ep2) 
                pl_3_loss = torch.mean(pl_3 * exp_variance_ep3)
                pl_4_loss = torch.mean(pl_4 * exp_variance_ep4)

                pl_loss = (pl_1_loss + pl_2_loss + pl_3_loss + pl_4_loss)/4

                consistency_weight_a = get_current_consistency_weight_a(iter_num // (iter_per_epoch * 2))
                consistency_weight_b = get_current_consistency_weight_b(iter_num // (iter_per_epoch * 2))

                loss = supervised_loss + consistency_weight_a*pl_loss + consistency_weight_b*l_uncert 
                                
                optimizer_1.zero_grad()
                
                loss.backward()

                optimizer_1.step()
                running_train_loss += loss.item()
                running_train_ce_loss += total_loss_ce.item()
                running_train_dice_loss += total_loss_dice.item()
                running_train_pl_loss += pl_loss.item()
                running_uncertainity_loss += l_uncert.item()
                running_ep1_loss += loss_ce_ep1.item()
                running_ep2_loss += loss_ce_ep2.item()
                running_ep3_loss += loss_ce_ep3.item()
                running_ep4_loss += loss_ce_ep4.item()
                running_train_iou += mIoU(outputs_ep1, labels_S1)
                running_train_dice += mDice(outputs_ep1, labels_S1)
                
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']

                iter_num = iter_num + 1

            epoch_loss = (running_train_loss) / (iter_per_epoch)
            epoch_ce_loss = (running_train_ce_loss) / (iter_per_epoch)
            epoch_dice_loss = (running_train_dice_loss) / (iter_per_epoch)
            epoch_pl_loss = (running_train_pl_loss) / (iter_per_epoch)
            epoch_iou = (running_train_iou) / (iter_per_epoch)
            epoch_dice = (running_train_dice) / (iter_per_epoch)
            epoch_ep1_loss = (running_ep1_loss) / (iter_per_epoch)
            epoch_ep2_loss = (running_ep2_loss) / (iter_per_epoch)
            epoch_ep3_loss = (running_ep3_loss) / (iter_per_epoch)
            epoch_ep4_loss = (running_ep4_loss) / (iter_per_epoch)

            epoch_uncertainity_loss = running_uncertainity_loss / (iter_per_epoch)

            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
            self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, epoch)

            self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)

            self.logger.info('Train md-loss: {}'.format(epoch_ep1_loss))
            self.writer.add_scalar('Train/mdloss', epoch_ep1_loss, epoch)
            self.logger.info('Train aux1-loss: {}'.format(epoch_ep2_loss))
            self.writer.add_scalar('Train/aux1', epoch_ep2_loss, epoch)
            self.logger.info('Train aux2-loss: {}'.format(epoch_ep3_loss))
            self.writer.add_scalar('Train/aux2', epoch_ep3_loss, epoch)
            self.logger.info('Train aux3-loss: {}'.format(epoch_ep4_loss))
            self.writer.add_scalar('Train/aux3', epoch_ep4_loss, epoch)

            self.logger.info('Train PL-loss: {}'.format(epoch_pl_loss))
            self.writer.add_scalar('Train/PL-Loss', epoch_pl_loss, epoch)

            self.logger.info('Train uncertainty: {}'.format(epoch_uncertainity_loss))
            self.writer.add_scalar('Train/Uncertainty', epoch_uncertainity_loss, epoch)

            self.logger.info('Train IoU: {}'.format(epoch_iou))
            self.writer.add_scalar('Train/IoU', epoch_iou, epoch)
            self.logger.info('Train Dice: {}'.format(epoch_dice))
            self.writer.add_scalar('Train/Dice', epoch_dice, epoch)

            self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/consis_weight a', consistency_weight_a, epoch)
            self.writer.add_scalar('info/consis_weight b', consistency_weight_b, epoch)
            torch.cuda.empty_cache()

            self.model.eval()
            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images = images.to(device)
                    gts = gts.to(device)
                    
                    prediction_1, _, _, _ = self.model(images)

                loss_ce_1 = ce_loss(prediction_1, gts.long())
                loss_dice_1 = 1 - mDice(prediction_1, gts)
                val_loss = 0.5 * (loss_dice_1 + loss_ce_1)

                running_val_loss += val_loss.item()
                running_val_ce_loss += loss_ce_1.item()
                running_val_dice_loss += loss_dice_1.item()

                running_val_iou_1 += mIoU(prediction_1, gts)
                running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts)
                 
            epoch_loss_val = running_val_loss / len(val_loader)
            epoch_ce_loss_val = running_val_ce_loss / len(val_loader)
            epoch_dice_loss_val = running_val_dice_loss / len(val_loader)
            epoch_dice_val_1 = running_val_dice_1 / len(val_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(val_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(val_loader)

            scheduler_1.step(epoch_dice_val_1)
            
            self.logger.info('Val loss: {}'.format(epoch_loss_val))
            self.writer.add_scalar('Validation/loss', epoch_loss_val, epoch)
            self.logger.info('Val CE loss: {}'.format(epoch_ce_loss_val))
            self.writer.add_scalar('Validation/ce-loss', epoch_ce_loss_val, epoch)
            self.logger.info('Val Dice loss: {}'.format(epoch_dice_loss_val))
            self.writer.add_scalar('Validation/dice-loss', epoch_dice_loss_val, epoch)

            self.logger.info('Validation dice : {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Validation/mDice', epoch_dice_val_1, epoch)
            self.logger.info('Validation IoU : {}'.format(epoch_iou_val_1))
            self.writer.add_scalar('Validation/mIoU', epoch_iou_val_1, epoch)
            self.logger.info('Validation Accuracy : {}'.format(epoch_accuracy_val_1))
            self.writer.add_scalar('Validation/Accuracy', epoch_accuracy_val_1, epoch)
            
            mdice_coeff_1 =  epoch_dice_val_1

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1
                        
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
                torch.save(state_1, Checkpoints_Path + '/UASS_' + dataset_name + '.pth')
             
            self.logger.info(
                'current best dice coef: model: {}'.format(self.best_dice_coeff_1))
            self.logger.info('current patience :{}'.format(self.patience))
            print('Current lr:', lr_)
            print('pseudo mix weight:', lbl_weight)            
            print('==============================================')
            print('==============================================')
            print('==============================================')
if __name__ == '__main__':
    train_network = Network()
    train_network.run()