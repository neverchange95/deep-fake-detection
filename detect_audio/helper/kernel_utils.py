import functools
import torch
import torchaudio
import csv
import torch.nn as nn
import numpy as np

from pathlib import Path
from typing import Union, List, Optional, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import apply_codec
from module.lfcc import LFCC
from helper.utils import find_wav_files

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

class AudioDataset(Dataset):
    """Torch dataset to load data from a provided directory.

    Args:
        directory_or_path_list: Path to the directory containing wav files to load. Or a list of paths.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        directory_or_path_list: Union[Union[str, Path], List[Union[str, Path]]],
        sample_rate: int = 16_000,
        amount: Optional[int] = None,
        normalize: bool = True,
        trim: bool = True,
        phone_call: bool = False,
    ) -> None:
        super().__init__()

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.phone_call = phone_call

        if isinstance(directory_or_path_list, list):
            paths = directory_or_path_list
        elif isinstance(directory_or_path_list, Path) or isinstance(
            directory_or_path_list, str
        ):
            directory = Path(directory_or_path_list)
            if not directory.exists():
                raise IOError(f"Directory does not exists: {self.directory}")

            paths = find_wav_files(directory)
            if paths is None:
                raise IOError(f"Directory did not contain wav files: {self.directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!"
            )

        if amount is not None:
            paths = paths[:amount]

        self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path.as_posix(), normalize=self.normalize)

        # resamplling
        if sample_rate != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )

        if self.trim:
            (
                waveform_trimmed,
                sample_rate_trimmed,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE
            )

            if waveform_trimmed.size()[1] > 0:
                waveform = waveform_trimmed
                sample_rate = sample_rate_trimmed

        if self.phone_call:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                effects=[
                    ["lowpass", "4000"],
                    [
                        "compand",
                        "0.02,0.05",
                        "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                        "-8",
                        "-7",
                        "0.05",
                    ],
                    ["rate", "8000"],
                ],
            )
            waveform = apply_codec(waveform, sample_rate, format="gsm")

        audio_path = str(path)

        return waveform, sample_rate, str(audio_path)

    def __len__(self) -> int:
        return len(self._paths)

class PadDataset(Dataset):
    def __init__(self, dataset: Dataset, cut: int = 64600, label=None):
        self.dataset = dataset
        self.cut = cut  # max 4 sec (ASVSpoof default)
        self.label = label

    def __getitem__(self, index):
        waveform, sample_rate, audio_path = self.dataset[index]
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]
        if waveform_len >= self.cut:
            if self.label is None:
                return waveform[: self.cut], sample_rate, str(audio_path)
            else:
                return waveform[: self.cut], sample_rate, str(audio_path), self.label
        # need to pad
        num_repeats = int(self.cut / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, : self.cut][0]

        if self.label is None:
            return padded_waveform, sample_rate, str(audio_path)
        else:
            return padded_waveform, sample_rate, str(audio_path), self.label

    def __len__(self):
        return len(self.dataset)


class TransformDataset(Dataset):
    """A generic transformation dataset.

    Takes another dataset as input, which provides the base input.
    When retrieving an item from the dataset, the provided transformation gets applied.

    Args:
        dataset: A dataset which return a (waveform, sample_rate)-pair.
        transformation: The torchaudio transformation to use.
        needs_sample_rate: Does the transformation need the sampling rate?
        transform_kwargs: Kwargs for the transformation.
    """

    def __init__(
        self,
        dataset: Dataset,
        transformation: Callable,
        needs_sample_rate: bool = False,
        transform_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self._dataset = dataset

        self._transform_constructor = transformation
        self._needs_sample_rate = needs_sample_rate
        self._transform_kwargs = transform_kwargs

        self._transform = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate, audio_path, label = self._dataset[index]

        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs
                )
            else:
                self._transform = self._transform_constructor(**self._transform_kwargs)

        return self._transform(waveform), sample_rate, str(audio_path), label

class DoubleDeltaTransform(torch.nn.Module):
    """A transformation to compute delta and double delta features.

    Args:
        win_length (int): The window length to use for computing deltas (Default: 5).
        mode (str): Mode parameter passed to padding (Default: replicate).
    """

    def __init__(self, win_length: int = 5, mode: str = "replicate") -> None:
        super().__init__()
        self.win_length = win_length
        self.mode = mode

        self._delta = torchaudio.transforms.ComputeDeltas(
            win_length=self.win_length, mode=self.mode
        )

    def forward(self, X):
        """
        Args:
             specgram (Tensor): Tensor of audio of dimension (..., freq, time).
        Returns:
            Tensor: specgram, deltas and double deltas of size (..., 3*freq, time).
        """
        delta = self._delta(X)
        double_delta = self._delta(delta)

        return torch.hstack((X, delta, double_delta))


def _build_preprocessing(
    directory_or_audiodataset: Union[Union[str, Path], AudioDataset],
    transform: torch.nn.Module,
    audiokwargs: dict = {},
    transformkwargs: dict = {},
) -> TransformDataset:
    """Generic function template for building preprocessing functions."""
    if isinstance(directory_or_audiodataset, AudioDataset) or isinstance(
        directory_or_audiodataset, PadDataset
    ):
        return TransformDataset(
            dataset=directory_or_audiodataset,
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    elif isinstance(directory_or_audiodataset, str) or isinstance(
        directory_or_audiodataset, Path
    ):
        return TransformDataset(
            dataset=AudioDataset(directory=directory_or_audiodataset, **audiokwargs),
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    else:
        raise TypeError("Unsupported type for directory_or_audiodataset!")

mfcc = functools.partial(_build_preprocessing, transform=torchaudio.transforms.MFCC)
lfcc = functools.partial(_build_preprocessing, transform=LFCC)

def double_delta(dataset: Dataset, delta_kwargs: dict = {}) -> TransformDataset:
    return TransformDataset(
        dataset=dataset,
        transformation=DoubleDeltaTransform,
        transform_kwargs=delta_kwargs,
    )

def load_directory(
    path: Union[Path, str],
    feature_fn: Callable,
    feature_kwargs: dict,
    test_size: float,
    use_double_delta: bool = True,
    phone_call: bool = False,
    pad: bool = False,
    label: Optional[int] = None,
    amount_to_use: Optional[int] = None,
) -> Tuple[TransformDataset, TransformDataset]:
    """Load all wav files from directory, apply the feature transformation
    and split into test/train.

    Args:
        path (Union[Path, str]): Path to directory.
        feature_fn (Callable): This is assumed to be mfcc or lfcc function.
        feature_fn (dict): Kwargs for the feature_fn.
        test_size (float): Ratio of train/test split.
        use_double_delta (bool): Additionally calculate delta and double delta features (Default True)?
        amount_to_use (Optional[int]): If supplied, limit data.
    """
    paths = find_wav_files(path)
    if paths is None:
        raise IOError(f"Could not load files from {path}!")

    if amount_to_use is not None:
        paths = paths[:amount_to_use]

    print(f"Loading data from {path}...!")

    test_dataset = AudioDataset(paths, phone_call=phone_call)
    if pad:
        test_dataset = PadDataset(test_dataset, label=label)

    if feature_fn is None:
        return test_dataset

    dataset_test = feature_fn(
        directory_or_audiodataset=test_dataset,
        transformkwargs=feature_kwargs,
    )
    if use_double_delta:
        dataset_test = double_delta(dataset_test)

    return dataset_test

def predict_on_audio(
        device,
        batch_size,
        model: nn.Module,
        dataset_test: Dataset,
        output_file: Union[str, Path] = None,
        checkpoint: dict = None,
) -> None:
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")

        # evaluation
        model.eval()
        num_correct = 0.0
        num_total = 0.0

        y_pred = []
        filenames = []

        for batch_x, _, filenames_batch, batch_y in test_loader:
            # get actual batch size
            curr_batch_size = batch_x.size(0)
            num_total += curr_batch_size
            # get batch input x
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            # make batch label y a vector
            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(device)
            filenames.extend(filenames_batch)
            # forward / inference
            batch_out = model(batch_x)
            # get probabilities from sigmoid output
            batch_pred = torch.sigmoid(batch_out)
            y_pred.append(batch_pred.clone().detach().cpu())
        
        # get all predictions
        y_pred: np.ndarray = torch.cat(y_pred, dim=0).numpy()

   # Create CSV file
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prediction"])
            for filename, pred_probs in zip(filenames, y_pred):
                writer.writerow([filename, pred_probs.item()])

        return None