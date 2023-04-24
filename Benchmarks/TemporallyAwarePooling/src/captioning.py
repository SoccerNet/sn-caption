import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetCaptions, PredictionCaptions, collate_fn_padd
from model import Video2Caption
from train import trainer, test_captioning, validate_captioning

from utils import valid_probability

import wandb


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
        dataset_Valid = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
        dataset_Valid_metric  = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
    dataset_Test  = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][0].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    model = Video2Caption(vocab_size=dataset_Test.vocab_size, weights=args.load_weights, input_size=args.feature_dim,
                  window_size=args.window_size_caption, 
                  vlad_k = args.vlad_k,
                  framerate=args.framerate,
                  pool=args.pool,
                  num_layers=args.num_layers,
                  teacher_forcing_ratio=args.teacher_forcing_ratio, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd)


    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=0, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer("caption", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "caption","model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    # validate caption generation on groundtruth spots on multiple splits [test/challenge]
    for split in args.split_test:

        dataset_Test  = SoccerNetCaptions(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_test,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            )

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn_padd)

        results = validate_captioning(test_loader, model, args.model_name)
        if results is None:
            continue

        logging.info("Best Performance at end of training in generating captions")
        logging.info(f'| Bleu_1: {results["Bleu_1"]}')
        logging.info(f'| Bleu_2: {results["Bleu_2"]}')
        logging.info(f'| Bleu_3: {results["Bleu_3"]}')
        logging.info(f'| Bleu_4: {results["Bleu_4"]}')
        logging.info(f'| METEOR: {results["METEOR"]}')
        logging.info(f'| ROUGE_L: {results["ROUGE_L"]}')
        logging.info(f'| CIDEr: {results["CIDEr"]}')

        wandb.log({f"{k}_{split}_gt" : v for k, v in results.items()})


    return 

def dvc(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    dataset_Test  = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][0].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    model = Video2Caption(vocab_size=dataset_Test.vocab_size, weights=args.load_weights, input_size=args.feature_dim,
                  window_size=args.window_size_caption, 
                  vlad_k = args.vlad_k,
                  framerate=args.framerate,
                  pool=args.pool,
                  num_layers=args.num_layers,
                  teacher_forcing_ratio=args.teacher_forcing_ratio).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "caption","model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    # generate dense caption on multiple splits [test/challenge]
    for split in args.split_test:
        PredictionPath = os.path.join("models", args.model_name, f"outputs/{split}")
        dataset_Test  = PredictionCaptions(SoccerNetPath=args.SoccerNet_path, PredictionPath=PredictionPath, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size_caption)

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        results = test_captioning(test_loader, model, args.model_name)
        if results is None:
            continue

        logging.info("Best Performance at end of training in dense video captioning")
        logging.info(f'| Bleu_1_tight: {results["Bleu_1_tight"]}')
        logging.info(f'| Bleu_2_tight: {results["Bleu_2_tight"]}')
        logging.info(f'| Bleu_3_tight: {results["Bleu_3_tight"]}')
        logging.info(f'| Bleu_4_tight: {results["Bleu_4_tight"]}')
        logging.info(f'| METEOR_tight: {results["METEOR_tight"]}')
        logging.info(f'| ROUGE_L_tight: {results["ROUGE_L_tight"]}')
        logging.info(f'| CIDEr_tight: {results["CIDEr_tight"]}')
        logging.info(f'| Recall_tight: {results["Recall_tight"]}')
        logging.info(f'| Precision_tight: {results["Precision_tight"]}')

        logging.info(f'| Bleu_1_loose: {results["Bleu_1_loose"]}')
        logging.info(f'| Bleu_2_loose: {results["Bleu_2_loose"]}')
        logging.info(f'| Bleu_3_loose: {results["Bleu_3_loose"]}')
        logging.info(f'| Bleu_4_loose: {results["Bleu_4_loose"]}')
        logging.info(f'| METEOR_loose: {results["METEOR_loose"]}')
        logging.info(f'| ROUGE_L_loose: {results["ROUGE_L_loose"]}')
        logging.info(f'| CIDEr_loose: {results["CIDEr_loose"]}')
        logging.info(f'| Recall_loose: {results["Recall_loose"]}')
        logging.info(f'| Precision_loose: {results["Precision_loose"]}')

        wandb.log({f"{k}_{split}_pt" : v for k, v in results.items()})


if __name__ == '__main__':


    parser = ArgumentParser(description='SoccerNet-Caption: Captioning training', formatter_class=ArgumentDefaultsHelpFormatter)

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
    parser.add_argument('--window_size_caption', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vlad_k',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--min_freq',       required=False, type=int,   default=5, help='Minimum word frequency to the vocabulary for caption generation' )
    
    parser.add_argument('--teacher_forcing_ratio',  required=False, type=valid_probability,   default=1, help='Teacher forcing ratio to use' )
    parser.add_argument('--num_layers',  required=False, type=int,   default=2, help='Teacher forcing ratio to use' )
    parser.add_argument('--freeze_encoder',  required=False, type=bool, default=False)
    parser.add_argument('--pretrain',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--weights_encoder',  required=False, type=str, default=None)
    parser.add_argument('--first_stage',  required=False, type=str,  choices=["spotting", "caption"], default="spotting")

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
    project="NetVLAD-caption",
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
