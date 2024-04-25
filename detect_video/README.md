# Detect Deep-Fake-Videos

### Download the pretrained models:

```
sh download_models.sh
```

### Run the detector with default configuration:

```
python predict_videos.py
```

### Run the detector with own configuration:

```
python predict_videos.py --config /path/to/your/config.yml
```

### General infos:

- Videos for prediction must be stored in the `/data` folder in `.mp4` format
- Submissions are stored in the `submission.csv` file
- `config.yml` contains the configuration
  - `model_dir` = path to pretrained models
  - `pretrained_models` = filenames of the pretrained models
  - `video_dir` = path to the directory where the video files are located
  - `output_file` = filename where the submissions are stored
