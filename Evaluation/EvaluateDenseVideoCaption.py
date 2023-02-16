import argparse

from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')

    parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V2 dataset folder')
    parser.add_argument('--Predictions_path',   required=True, type=str, help='Path to the predictions folder' )
    parser.add_argument('--split',   required=False, type=str, default= "test", help='Set on which to evaluate the performances')
    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--prediction_file',   required=False, type=str, default="results_dense_captioning.json", help='prediction filename')
    parser.add_argument('--label_files',   required=False, type=str, default="Labels-caption.json", help='label filename')
    parser.add_argument('--include_SODA',required=False, action='store_true',  help='Perform testing only')

    args = parser.parse_args()

    print(evaluate(SoccerNet_path=args.SoccerNet_path, Predictions_path=args.Predictions_path, split=args.split, version=args.version, prediction_file=args.prediction_file, label_files=args.label_files, include_SODA=args.include_SODA))