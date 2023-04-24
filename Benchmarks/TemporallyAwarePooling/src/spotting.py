import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting
from model import Video2Spot
from train import trainer, test_spotting
from loss import NLLLoss

import wandb

def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)
        dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)
        dataset_Valid_metric  = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)
    dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][1].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    model = Video2Spot(weights=args.load_weights, input_size=args.feature_dim,
                  num_classes=dataset_Test.num_classes, window_size=args.window_size_spotting, 
                  vlad_k=args.vlad_k,
                  framerate=args.framerate, pool=args.pool, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        criterion = NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=0, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer("spotting", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "spotting", "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    for split in args.split_test:
        dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

        results = test_spotting(test_loader, model=model, model_name=args.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold)
        if results is None:
            continue

        a_mAP_tight = results["a_mAP_tight"]
        a_mAP_loose = results["a_mAP_loose"]
        a_mAP_medium = results["a_mAP_medium"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP tight: " +  str(a_mAP_tight))
        logging.info("a_mAP loose: " +  str(a_mAP_loose))
        logging.info("a_mAP_medium: " +  str(a_mAP_medium))
                
        wandb.log({f"{k}_{split}" : results[k] for k in ["a_mAP_tight", "a_mAP_loose", "a_mAP_medium"]})

    return 

if __name__ == '__main__':


    parser = ArgumentParser(description='SoccerNet-Caption: Spotting training', formatter_class=ArgumentDefaultsHelpFormatter)
    
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
    parser.add_argument('--window_size', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vlad_k',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
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
    project="NetVLAD-spotting",
    name=args.model_name
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
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
