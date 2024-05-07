# Detect Deep-Fake-Audios

### Best pretrained model:

```
models/pretrained/ShallowCNN_lfcc_I/best.pt
```

### Run the detector with default configuration:

```
python predict_audios.py
```

### Run the detector with own configuration:

```
python predict_audios.py --config /path/to/your/config.yml
```

### General infos:

- Audios for prediction must be stored in the `/data` folder.
- Submissions are stored in the `submission.csv` file
- `config.yml` contains the configuration
  - `model_dir` = path to pretrained models
  - `model_name` = modelname which should be used for prediction.
  - `feature_name` = featurename which should be used for prediction.
  - `audio_dir` = path to the directory where the audio files are located
  - `output_file` = filename where the submissions are stored
