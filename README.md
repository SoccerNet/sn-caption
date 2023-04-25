# SoccerNet Dense Video Captioning 

Bring your soccer game experience to life with Dense Video Captioning! This cutting-edge technology highlights the most exciting moments and adds captivating commentaries, immersing you in the action like never before.

The task of Dense Video Captioning consists in generating coherent caption describing soccer actions occured and localizing each caption by a timestamp. For that, you have access to 471 videos from soccer broadcast games available at two resolutions (720p and 224p) with captions. We also provide extracted features at 2 frames per second for an easier use, including the feature provided by the 2021 action spotting challenge winners, Baidu Research. The provided data also includes original comments and versions where referees, coaches, players, and teams have been anonymized or identified, as well as team lineups. The challenge set is composed of 42 separate games.

The [evaluation server](https://eval.ai/web/challenges/challenge-page/1947/overview) is already available for you to submit your predictions.

<p align="center"><img src="Images/logo-caption.jpg" width="640"></p>

## How to download the dataset

A [SoccerNet pip package](https://pypi.org/project/SoccerNet/) to easily download the data and the annotations is available. 

To install the pip package simply run:

<code>pip install SoccerNet</code>

Then use the API to downlaod the data of interest including annotations and features at 2fps:

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="caption-2023", split=["train","valid", "test","challenge"])
```

If you want to download the videos, you will need to fill a [NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get the password.

```
mySoccerNetDownloader.password = input("Password for videos?:\n")
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "video.ini"], split=["train","valid","test","challenge"])
```

## Our other Challenges

Check out our other challenges related to SoccerNet!
- [Action Spotting](https://github.com/SoccerNet/sn-spotting)
- [Replay Grounding](https://github.com/SoccerNet/sn-grounding)
- [Calibration](https://github.com/SoccerNet/sn-calibration)
- [Re-Identification](https://github.com/SoccerNet/sn-reid)
- [Tracking](https://github.com/SoccerNet/sn-tracking)
- [Jersey Number Recognition](https://github.com/SoccerNet/sn-jersey)

## Citation

Please cite our work if you use our dataset:
```
@article{Mkhallati2023SoccerNetCaption-arxiv,
	title = {{SoccerNet}-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries},
	author = {Mkhallati, Hassan and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc},
	journal = arxiv,
	volume = {abs/2304.04565},
	year = {2023},
	publisher = {arXiv},
	eprint = {2304.04565},
	keywords = {},
	eprinttype = {arXiv},
	doi = {10.48550/arXiv.2304.04565},
	url = {https://doi.org/10.48550/arXiv.2304.04565}
}
```

