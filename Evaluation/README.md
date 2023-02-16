# Evaluation

We provide evaluation functions directly integrated in our pip package (`pip install SoccerNet`) as well as an evaluation server on [EvalAI]().

We defined the following metric for Dense Video Captioning task. For each predicted and ground truth captions, we build a timewindow tolerance of 30 seconds centered on the spotting timestamp (15 seconds before and 15 seconds after). We compute standard metric for generated text between a predicted caption with any ground truth caption those timewindows overlaps. 

This metric can be derived by the metric introduced in ActivityNet Caption.  After computing the timewindow tolerance of 30 seconds, we use the same procedure with tIoU > 0.

We also implented the [SODA](https://fujiso.github.io/publications/ECCV2020_soda.pdf) metric. You can compute with <code>--include_SODA</code>.


## Ouput Format

To submit your results on EvalAI or to use the integreted function of the pip package, the predictions of the network have to be saved in a specific format, with a specific folder structure.

```
Results.zip
 - league
   - season
     - game full name
       - results_dense_captioning.json
```

### `results_dense_captioning.json`

For the action spotting task, each json file needs to be constructed as follows:

```json
{
    "UrlLocal": "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal",
    "predictions": [ # list of predictions
        {
            "gameTime": "1 - 0:31", # format: "{half} - {minutes}:{seconds}",
            "label": "comments", # Always set to comments,
            "position": "31500", # time in milliseconds,
            "half": "1", # half of the game
            "confidence": "0.006630070507526398", # confidence score for the spotting,
            "comment": "[PLAYER] ([TEAM]) is booked after bringing down an opponent. [REFEREE] had an easy-decision to make." # caption string
        },
        {
            "gameTime": "1 - 0:39",
            "label": "comments",
            "position": "39500",
            "half": "1",
            "confidence": "0.07358131557703018",
            "comment": "[PLAYER] ([TEAM]) is booked after bringing down an opponent. [REFEREE] had an easy-decision to make."
        },
        {
            "gameTime": "1 - 0:55",
            "label": "comments",
            "position": "55500",
            "half": "1",
            "confidence": "0.20939764380455017",
            "comment": "[PLAYER] ([TEAM]) is booked after bringing down an opponent. [REFEREE] had an easy-decision to make."
        },
        ...
    ]
}
```

## How to evaluate locally the performances on the testing set

### Spotting

```bash
python EvaluateDenseVideoCaption.py --SoccerNet_path /path/to/SoccerNet/ --Predictions_path /path/to/SoccerNet/outputs/
```

```python
from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate
results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                   split="test", version=2, prediction_file="results_dense_captioning.json", include_SODA=False)


```

## How to evaluate online the performances on the challenge

### Zip the results

```bash
cd /path/to/soccernet/outputs/
zip results_dense_captioning.zip */*/*/results_dense_captioning.json
```

### Visit [EvalAI](https://eval.ai/auth/login) to submit you zipped results
