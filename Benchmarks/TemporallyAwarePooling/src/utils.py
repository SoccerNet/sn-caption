from SoccerNet.utils import getListGames
from SoccerNet.Evaluation.utils import LoadJsonFromZip, AverageMeter, getMetaDataTask
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import os
from tqdm import tqdm
import zipfile
import json
import numpy as np
import glob
import argparse

def evaluate(SoccerNet_path, Predictions_path, prediction_file="results_spotting.json", split="test", version=2, framerate=2, metric="loose"):
    # evaluate the prediction with respect to some ground truth
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP
    list_games = getListGames(split=split, task="caption")
    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    label_files, num_classes, _, _ = getMetaDataTask("caption", "SoccerNet", version)

    for game in tqdm(list_games):

        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector
        label_half_1, label_half_2 = label2vector(labels, num_classes=num_classes, version=version, framerate=framerate)
        # print(version)
        # print(label_half_1)
        # print(label_half_2)



        # infer name of the prediction_file
        if prediction_file == None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path,"*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    # print(prediction_file)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        predictions_half_1, predictions_half_2 = predictions2vector(predictions, num_classes=num_classes, version=version, framerate=framerate)

        targets_numpy.append(label_half_1)
        targets_numpy.append(label_half_2)
        detections_numpy.append(predictions_half_1)
        detections_numpy.append(predictions_half_2)

        closest_numpy = np.zeros(label_half_1.shape)-1
        #Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_1[indexes[i],c]
        closests_numpy.append(closest_numpy)

        closest_numpy = np.zeros(label_half_2.shape)-1
        for c in np.arange(label_half_2.shape[-1]):
            indexes = np.where(label_half_2[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_2[indexes[i],c]
        closests_numpy.append(closest_numpy)


    if metric == "loose":
        deltas=np.arange(12)*5 + 5
    elif metric == "tight":
        deltas=np.arange(5)*1 + 1
    elif metric == "medium":
        deltas = np.array([30])
    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    
    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version==2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version==2 else None,
        "a_mAP_unshown": a_mAP_unshown if version==2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version==2 else None,
    }
    return results

def label2vector(labels, num_classes=17, framerate=2, version=2):


    vector_size = 90*60*framerate

    label_half1 = np.zeros((vector_size, num_classes))
    label_half2 = np.zeros((vector_size, num_classes))

    _, _, event_dict, _ = getMetaDataTask("caption", "SoccerNet", version)

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = framerate * ( seconds + 60 * minutes ) 

        if event not in event_dict:
            continue
        label = event_dict[event]

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size-1)
            label_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            label_half2[frame][label] = value

    return label_half1, label_half2

def predictions2vector(predictions, num_classes=17, version=2, framerate=2):

    vector_size = 90*60*framerate

    prediction_half1 = np.zeros((vector_size, num_classes))-1
    prediction_half2 = np.zeros((vector_size, num_classes))-1

    _, _, event_dict, _ = getMetaDataTask("caption", "SoccerNet", version)

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * ( time/1000 ))

        if event not in event_dict:
            continue
        label = event_dict[event]

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size-1)
            prediction_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            prediction_half2[frame][label] = value

    return prediction_half1, prediction_half2


def valid_probability(value):
    fvalue = float(value)
    if fvalue <= 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid probability between 0 and 1")
    return fvalue