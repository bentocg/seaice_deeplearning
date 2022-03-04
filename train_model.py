import torch
from argparse import ArgumentParser
from torchvision.models.resnet import resnet34
import segmentation_models_pytorch as smp
from utils.training.training_manager import Trainer
from utils.training.utility import get_model_stats, seed_all
import os


def parse_args():
    parser = ArgumentParser("Inputs for temple classification pipeline")
    parser.add_argument("--training_set", "-t", type=str, help="path to training set")
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=64,
        help="batch size for training and validation",
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        type=float,
        default=0.0005,
        help="learning rate for training",
    )
    parser.add_argument(
        "--patch_size", "-p", type=int, default=256, help="patch size for images"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=25, help="number of epochs for training"
    )
    parser.add_argument(
        "--segmentation",
        "-s",
        type=int,
        default=0,
        help="switch for segmentation" " / classification models",
    )
    parser.add_argument(
        "--finetune",
        "-f",
        type=int,
        default=0,
        help="switch for segmentation models "
        "to finetune encoder from classification or"
        "train from scratch",
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--device_id", "-d", type=int, default=0, help="device id for cuda GPU"
    )
    parser.add_argument(
        "--autoresume",
        "-a",
        type=int,
        default=1,
        help="whether to autoresume training from last epoch",
    )
    parser.add_argument(
        "--tsets",
        "-z",
        type=str,
        default="hand",
        help="which training sets are used for training",
    )
    parser.add_argument(
        "--augmentation_mode",
        "-g",
        type=str,
        default="simple",
        help="what kind of data augmentation to be used during training",
    )
    parser.add_argument(
        "--neg_to_pos_ratio",
        "-n",
        type=float,
        default=1.0,
        help="number of negative samples for every positive sample in minibatches",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )
    parser.add_argument(
        "--criterion", "-c", type=str, default="BCE", help="loss function for training"
    )
    parser.add_argument(
        "--save_output",
        "-u",
        type=int,
        default=0,
        help="switch to store output to tensorboard",
    )
    return parser.parse_args()


def main():
    # load arguments
    args = parse_args()
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    state_dict = None

    # set seed
    seed_all(args.random_seed)

    # define model
    if args.segmentation:
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None, in_channels=1)

        # update with pretrained weights from classification
        if args.finetune:

            model_stats = get_model_stats()
            chosen = model_stats.iloc[:50].sample(1).iloc[0]
            chosen_idx = chosen.name
            chosen_model = chosen.model_name
            weights = torch.load(
                f"checkpoints/{chosen_model}", map_location=torch.device(device)
            )["state_dict"]
            model.load_state_dict(weights)

            # # find best_weights for current patch size

            # best = 'Resnet34_512_best.pth'

            # # load pretrained dict
            # pretrained_dict = torch.load(f'checkpoints/{best}',
            #                              map_location=torch.device(device))['state_dict']
            # pretrained_dict = {f'encoder.{k}': v for k, v in pretrained_dict.items()}
            # model_dict = model.state_dict()

            # # 1. filter out unnecessary keys
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # # 2. overwrite entries in the existing state dict
            # model_dict.update(pretrained_dict)

            # # 3. load the new state dict
            # model.load_state_dict(model_dict)
        if args.finetune == 0:
            ft = "scratch"
        else:
            ft = "finetuned" + str(chosen_idx)
        model_name = (
            f"UnetResnet34_{args.patch_size}_{args.learning_rate}_{args.batch_size}_"
            f"{ft}_tsets_{args.tsets}_"
            f"aug_{args.augmentation_mode}_ratio_{args.neg_to_pos_ratio}_loss_{args.criterion}"
        )

    else:
        model = resnet34(num_classes=1)
        model_name = (
            f"Resnet34_{args.patch_size}_{args.learning_rate}_{args.batch_size}"
            f"_tsets_{args.tsets}_"
            f"aug_{args.augmentation_mode}_ratio_{args.neg_to_pos_ratio}_loss_{args.criterion}"
        )

    # see if a checkpoint for this model already exists, load weights if it does
    checkpoint = f"checkpoints/{model_name}_last.pth"
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))
        print(f"Resuming from epoch {checkpoint['epoch']}")

        # skip past epochs and reload weights
        state_dict = checkpoint

    # start training
    tsets = tuple(args.tsets.split("_"))
    model_trainer = Trainer(
        model,
        device=device,
        patch_size=args.patch_size,
        batch_size=(args.batch_size, args.batch_size * 2),
        epochs=args.epochs,
        data_folder=args.training_set,
        lr=float(args.learning_rate),
        model_name=model_name,
        segmentation=args.segmentation,
        state_dict=state_dict,
        tsets=tsets,
        neg_to_pos_ratio=float(args.neg_to_pos_ratio),
        augmentation_mode=args.augmentation_mode,
        num_workers=args.num_workers,
        loss=args.criterion,
        save_output=args.save_output,
    )
    print(f"started training {model_name}")
    model_trainer.start()


if __name__ == "__main__":
    main()
