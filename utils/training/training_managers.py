from utils.training import Meter, epoch_log
from utils.loss_functions import MixedLoss, FocalLoss
from utils.data_processing import provider

import torch.optim as optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import time


class TrainerSegmentation(object):
    """This class takes care of training and validation of our segmentation models"""

    def __init__(self, model, model_name, device="cuda:0", batch_size=(64, 128), patch_size=256, epochs=20,
                 lr=1e-3, patience=3, data_folder='training_set_synthetic'):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 0
        self.batch_size = {'train': batch_size[0], 'val': batch_size[1]}
        self.lr = lr
        self.num_epochs = epochs
        self.best_dice = 0.0
        self.phases = ["train", "val"]
        self.device = device
        self.model_name = model_name
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = MixedLoss(10.0, 2.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=patience, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=1,
                total_folds=5,
                segmentation=True,
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

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, _, targets = batch
            loss, outputs = self.forward(images, targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(targets, outputs)
        epoch_loss = running_loss / (total_batches * batch_size)
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        torch.cuda.empty_cache()
        return dice

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_dice": self.best_dice,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_dice = self.iterate(epoch, "val")
            self.scheduler.step(val_dice)
            if val_dice > self.best_dice:
                print("******** New optimal found, saving state ********")
                state["best_dice"] = self.best_dice = val_dice
                torch.save(state, f"checkpoints/{self.model_name}_dice-{self.best_dice}_epoch-{epoch}.pth")
            print()


class TrainerClassification(object):
    """This class takes care of training and validation of our classification models"""

    def __init__(self, model, model_name, device="cuda:0", batch_size=(64, 128), patch_size=256, epochs=20,
                 lr=1e-3, patience=3, data_folder='training_set_classification'):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 0
        self.batch_size = {'train': batch_size[0], 'val': batch_size[1]}
        self.lr = lr
        self.num_epochs = epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = device
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = FocalLoss(2.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=patience, verbose=True)
        self.net = self.net.to(self.device)
        self.model_name = model_name
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=1,
                total_folds=5,
                segmentation=False,
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

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.flatten().to(self.device)
        outputs = self.net(images).flatten()
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
        epoch_loss = running_loss / total_batches
        self.losses[phase].append(epoch_loss)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, f"checkpoints/{self.model_name}_loss-{self.best_loss}_epoch-{epoch}.pth")
            print()