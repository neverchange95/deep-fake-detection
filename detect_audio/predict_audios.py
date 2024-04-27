import argparse
import yaml
import torch

from typing import Tuple, Dict
from torchinfo import summary
from helper.kernel_utils import lfcc, mfcc, load_directory, predict_on_audio
from models.cnn import ShallowCNN
from models.lstm import SimpleLSTM, WaveLSTM
from models.mlp import MLP
from models.rnn import WaveRNN
from models.tssd import TSSD
from pathlib import Path

# all feature classnames
FEATURE_NAMES: Tuple[str] = ("wave", "lfcc", "mfcc")

# all model classnames
MODEL_NAMES: Tuple[str] = (
    "MLP",
    "WaveRNN",
    "WaveLSTM",
    "SimpleLSTM",
    "ShallowCNN",
    "TSSD",
)

# all model keyword arguments
KWARGS_MAP: Dict[str, dict] = {
    "SimpleLSTM": {
        "lfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
        "mfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
    },
    "ShallowCNN": {
        "lfcc": {"in_features": 1, "out_dim": 1},
        "mfcc": {"in_features": 1, "out_dim": 1},
    },
    "MLP": {
        "lfcc": {"in_dim": 40 * 972, "out_dim": 1},
        "mfcc": {"in_dim": 40 * 972, "out_dim": 1},
    },
    "TSSD": {
        "wave": {"in_dim": 64600},
    },
    "WaveRNN": {
        "wave": {"num_frames": 10, "input_length": 64600, "hidden_size": 500},
    },
    "WaveLSTM": {
        "wave": {
            "num_frames": 10,
            "input_len": 64600,
            "hidden_dim": 30,
            "out_dim": 1,
        }
    },
}

if __name__ == '__main__':
    # prepare argument parser
    parser = argparse.ArgumentParser('Predict test audios')
    arg = parser.add_argument
    arg('--config', type=str, default='config.yml', help='path to config.yml file')
    args = parser.parse_args()

    # read config from .yml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # get config
    model_dir = config['model_dir']
    model_name = config['model_name']
    feature_name = config['feature_name']
    audio_dir = config['audio_dir']
    output_file = config['output_file']

    # check if feature and model is aviable
    feature_name = feature_name.lower()
    assert feature_name in FEATURE_NAMES
    assert model_name in MODEL_NAMES

    # get feature transformation function
    feature_fn = None if feature_name == 'wave' else eval(feature_name)
    assert feature_fn in (None, lfcc, mfcc)
    
    # get model constructor
    Model = eval(model_name)
    assert Model in (SimpleLSTM, ShallowCNN, WaveLSTM, MLP, TSSD, WaveRNN)

    model_kwargs: dict = KWARGS_MAP.get(model_name).get(feature_name)
    if model_kwargs is None:
        raise ValueError(f"model_kwargs not found for {model_name} and {feature_name}")
    
    # check if CUDA (Nvidia) or MPS (Apple Sillicon) is aviable
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs.update({"device": device})
    print(f"use device: {device}")

    dataset_test = load_directory(
        path=audio_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=0.0,
        use_double_delta=True,
        phone_call=False,
        pad=True,
        label=0,
        amount_to_use=None
    )

    print("Predicting {} audios".format(len(dataset_test)))

    model = Model(**model_kwargs).to(device)
    batch_size = 32
    input_size = (
        (batch_size, 64600) if feature_name == "wave" else (batch_size, 40, 972)
    )
    model_stats = summary(model, input_size, verbose=0)

    # load pretrained models
    model_path = Path(model_dir + '/pretrained/ShallowCNN_lfcc_I/best.pt')
    if model_path.is_file():
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        ckpt = None
    
    predict_on_audio(device=device, batch_size=batch_size, model=model,
                     dataset_test=dataset_test, output_file=output_file, 
                     checkpoint=ckpt)