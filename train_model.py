import torch
from argparse import ArgumentParser
from torchvision.models.resnet import resnet34
import segmentation_models_pytorch as smp
from utils.training.training_manager import Trainer
from utils.training.utility import seed_all
import os
import re


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
    state_dict = None

    # set seed
    seed_all(args.random_seed)

    # define model
    if args.segmentation:
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

        # update with pretrained weights from classification
        if args.finetune:

            # find best_weights for current patch size
            model_names = [ele for ele in os.listdir('checkpoints') if ele.startswith(f'Resnet34_{args.patch_size}')
                           and 'last' not in ele]
            best = sorted(model_names,
                          key=lambda x: float(re.search('dice-[0-9]{1}.[0-9]+', x).group(0).split('-')[-1]))[-1]

            # load pretrained dict
            pretrained_dict = torch.load(f'checkpoints/{best}',
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

    # see if a checkpoint for this model already exists, load weights if it does
    checkpoint = f"checkpoints/{model_name}_last.pth"
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint,
                                map_location=torch.device(device))
        print(f"Resuming from epoch {checkpoint['epoch']}")

        # skip past epochs and reload weights
        state_dict = checkpoint

    # start training
    model_trainer = Trainer(model, device=device, patch_size=args.patch_size,
                            batch_size=(args.batch_size, args.batch_size * 2), epochs=args.epochs,
                            data_folder=args.training_set, lr=args.learning_rate,
                            model_name=model_name, segmentation=args.segmentation,
                            state_dict=state_dict)
    model_trainer.start()


if __name__ == '__main__':
    main()
