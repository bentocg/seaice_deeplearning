from utils.training import Meter, epoch_log
from utils.loss_functions import MixedLoss, FocalLoss
from utils.data_processing import provider
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import time
import os


class Trainer(object):
    """This class takes care of training and validation of our segmentation models"""

    def __init__(self, model, model_name, device="cuda:0", batch_size=(64, 128), patch_size=256, epochs=20,
                 lr=1e-3, patience=3, data_folder='training_set_synthetic', segmentation=True,
                 state_dict=None):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 0
        self.batch_size = {'train': batch_size[0], 'val': batch_size[1]}
        self.lr = lr
        self.num_epochs = epochs
        self.start_epoch = 0
        self.best_iou = 0.0
        self.best_dice = 0.0
        self.phases = ["train", "val"]
        self.device = device
        self.segmentation = segmentation
        self.model_name = model_name
        self.global_step = 0
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        if self.segmentation:
            self.criterion = MixedLoss(10.0, 2.0)
        else:
            self.criterion = FocalLoss(2.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        if state_dict:
            self.start_epoch = state_dict['epoch']
            self.global_step = state_dict['global_step']
            self.best_iou = state_dict['best_iou']
            self.best_dice = state_dict['best_dice']
            self.net.load_state_dict(state_dict['state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.state = state_dict
        else:
            self.state = {
                "epoch": 0,
                "global_step": 0,
                "best_dice": 0,
                "best_iou": 0,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=patience, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=1,
                total_folds=5,
                segmentation=self.segmentation,
                data_folder=data_folder,
                df_path=f'{data_folder}/labels.csv',
                phase=phase,
                size=patch_size,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.writer = SummaryWriter(comment=self.model_name)
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(segmentation=self.segmentation)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch[0], batch[-1]
            loss, outputs = self.forward(images, targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.global_step % 10 == 0:
                self.writer.add_scalar(f'Loss/{phase}', loss.item(), self.global_step)
            running_loss += loss.item()
            meter.update(targets, outputs)
            self.global_step += 1
        epoch_loss = running_loss / (total_batches * batch_size)
        dice, iou = epoch_log(epoch_loss, meter)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        self.writer.add_scalar(f'Dice/{phase}', dice, epoch)
        self.writer.add_scalar(f'IoU/{phase}', iou, epoch)
        p = torch.tensor([1 / len(images)] * len(images))
        idcs = p.multinomial(min(10, len(images)))
        images = self.inv_normalize(images)

        self.writer.add_images(f'input_images/{phase}', images[idcs], epoch)
        if self.segmentation:
            self.writer.add_images(f'masks_true/{phase}', targets[idcs] * 255, epoch)
            masks_pred = torch.sigmoid(outputs[idcs])
            self.writer.add_images(f'masks_pred/{phase}', (masks_pred > 0.5).float(), epoch)
        else:
            self.writer.add_scalar(f'true_label/{phase}', targets[idcs], epoch)
            label_pred = torch.sigmoid(outputs[idcs])
            self.writer.add_scalar(f'predicted_label/{phase}', (label_pred > 0.5).float(), epoch)

        torch.cuda.empty_cache()
        self.state['epoch'] = epoch
        self.state['global_step'] = self.global_step
        return dice, iou

    def start(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.iterate(epoch, "train")

            with torch.no_grad():
                val_dice, val_iou = self.iterate(epoch, "val")
            self.scheduler.step(val_dice)
            torch.save(self.state,  f"checkpoints/{self.model_name})_last.pth")
            if val_dice > self.best_dice:
                # remove previous best checkpoint for this model
                prev = [f'checkpoints/{ele}' for ele in os.listdir('checkpoints') if
                        ele.startswith(f'{self.model_name}_dice')]
                for ele in prev:
                    os.remove(ele)

                print("******** New optimal found, saving state ********")
                self.state["best_dice"] = self.best_dice = val_dice
                self.state["best_iou"] = self.best_iou = val_iou
                torch.save(self.state, f"checkpoints/{self.model_name}_dice-{self.best_dice}_iou-{self.best_iou}_epoch-{epoch}.pth")
            print()
        quit()
