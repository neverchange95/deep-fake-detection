import argparse
import os
import re
import time
import yaml

import torch
import pandas as pd

from helper.kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from helper.classifiers import DeepFakeClassifier

if __name__ == '__main__':
    # prepare argument parser
    parser = argparse.ArgumentParser('Predict test videos')
    arg = parser.add_argument
    arg('--config', type=str, default='config.yml', help='path to config.yml file')
    args = parser.parse_args()

    # read config from .yml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # get config
    model_dir = config['model_dir']
    model_names = config['pretrained_models']
    video_dir = config['video_dir']
    output_file = config['output_file']
    
    # load pretrained models
    models = []
    model_paths = [os.path.join(model_dir, name) for name in model_names]

    # check if CUDA (Nvidia) or MPS (Apple Sillicon) is aviable
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"use device: {device}")

    # load models on device
    for path in model_paths:
        # load models
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        models.append(model.half())

    # read video and extract faces
    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy
    stime = time.time()

    # predict videos
    videos = sorted([x for x in os.listdir(video_dir) if x[-4:] == ".mp4"])
    print("Predicting {} videos".format(len(videos)))
    predictions = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                       strategy=strategy, frames_per_video=frames_per_video, videos=videos,
                                       num_workers=6, test_dir=video_dir, device=device)
    submission_df = pd.DataFrame({'filename': videos, "label": predictions})
    submission_df.to_csv(output_file, index=False)
    print("Elapsed:", time.time() - stime)