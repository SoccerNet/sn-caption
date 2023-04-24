import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import torch
import wandb
import logging
import time
from datetime import datetime
from utils import valid_probability
import spotting
import captioning




if __name__ == '__main__':


    parser = ArgumentParser(description='SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/path/to/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_TF2.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="NetVLAD++",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Number of chunks per epoch' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vlad_k',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )
    parser.add_argument('--min_freq',       required=False, type=int,   default=5, help='Minimum word frequency to the vocabulary for caption generation' )
    parser.add_argument('--teacher_forcing_ratio',  required=False, type=valid_probability,   default=1, help='Teacher forcing ratio to use' )

    parser.add_argument('--first_stage',  required=False, type=str,  choices=["spotting", "caption"], default="spotting")
    parser.add_argument('--window_size_spotting', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--window_size_caption', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--freeze_encoder',  required=False, action='store_true',  help='Perform testing only')
    parser.add_argument('--pretrain',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--weights_encoder',  required=False, type=str, default=None)
    parser.add_argument('--num_layers',  required=False, type=int, default=2)
    

    parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))

    run = wandb.init(
    project="DVC-SoccerNet",
    name=args.model_name,
    )

    wandb.config.update(args)

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()

    frozen = args.freeze_encoder
    if args.first_stage == "spotting":
        logging.info('Starting main function')
        spotting.main(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')

        logging.info('Starting main function')
        args.weights_encoder = f"models/{args.model_name}/spotting/model.pth.tar" if args.pretrain else None
        captioning.main(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
    else:
        logging.info('Starting main function')
        captioning.main(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')

        logging.info('Starting main function')
        args.weights_encoder = f"models/{args.model_name}/caption/model.pth.tar" if args.pretrain else None
        spotting.main(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
    
    args.weights_encoder = None
    captioning.dvc(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')