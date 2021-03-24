import torch
from argparse import ArgumentParser
from torchvision.models.resnet import resnet34
import segmentation_models_pytorch as smp
from utils.training.training_manager import Trainer
from utils.training.utility import seed_all
from torch import optim
import os


def parse_args():
    parser = ArgumentParser('Inputs for temple classification pipeline')
    parser.add_argument('--training_set', '-t', type=str, help='path to training set')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size for training and validation')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--patch_size', '-p', type=int, default=256, help='patch size for images')
    parser.add_argument('--epochs', '-e', type=int, default=25, help='number of epochs for training')
    parser.add_argument('--segmentation', '-s', type=int, default=0, help='switch for segmentation'
                                                                          ' / classification models')
    parser.add_argument('--finetune', '-f', type=int, default=0, help='switch for segmentation models '
                                                                      'to finetune encoder from classification or'
                                                                      'train from scratch')
    parser.add_argument('--random_seed', '-r', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--device_id', '-d', type=int, default=0, help='device id for cuda GPU')
    parser.add_argument('--autoresume', '-a', type=int, default=1, help='whether to autoresume training from last epoch')
    return parser.parse_args()


def main():
    # load arguments
    args = parse_args()
    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs
    start_epoch = 0

    # set seed
    seed_all(args.random_seed)

    # define model
    if args.segmentation:
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

        if args.finetune:
            # update with pretrained weights from classification
            pretrained_dict = torch.load(f'checkpoints/Resnet34_{args.patch_size}_best.pth',
                                         map_location=torch.device(device))['state_dict']
            pretrained_dict = {f'encoder.{k}': v for k, v in pretrained_dict.items()}
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            model.load_state_dict(model_dict)

        model_name = f"UnetResnet34_{args.patch_size}_{args.learning_rate}_{args.batch_size}_" \
                     f"{'finetuned' if args.finetune else 'scratch'}"

    else:
        model = resnet34(num_classes=1)
        model_name = f"Resnet34_{args.patch_size}_{args.learning_rate}_{args.batch_size}"

    # start optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # see if a checkpoint for this model already exists, load weights if it does
    checkpoint = f"checkpoints/{model_name}_last.pth"
    if os.path.exists(checkpoint):
        checkpoint = torch.load(f'checkpoints/{checkpoint}',
                                map_location=torch.device(device))
        print(f"Resuming from epoch {checkpoint['epoch']}")

        # skip past epochs and reload weights
        start_epoch = checkpoint['epoch'] + 1
        epochs -= start_epoch
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # start training
    model_trainer = Trainer(model, optimizer, device=device, patch_size=args.patch_size,
                            batch_size=(args.batch_size, args.batch_size * 2), epochs=epochs,
                            data_folder=args.training_set,
                            model_name=model_name, segmentation=args.segmentation,
                            start_epoch=start_epoch)
    model_trainer.start()


if __name__ == '__main__':
    main()
