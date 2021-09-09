from typing import ClassVar, Tuple
from simple_parsing.helpers.independent_subparsers import add_independent_subparsers

from dataclasses import dataclass
from simple_parsing import ArgumentParser
from pathlib import Path
import os


@dataclass
class DatasetConfig:
    """ Configuration options for the dataset. """

    image_size: ClassVar[Tuple[int, int, int]]

    # Number of samples to keep. (-1 to keep all samples).
    n_samples: int = -1
    # Wether to shuffle the dataset.
    shuffle: bool = True


@dataclass
class MnistConfig(DatasetConfig):
    """ Configuration options for the MNIST dataset. """

    image_size: ClassVar[Tuple[int, int, int]] = (28, 28, 1)
    n_samples: int = 10_000  # some random number just for sake of illustration.
    shuffle: bool = True


@dataclass
class ImageNetConfig(DatasetConfig):
    """ Configuration options for the ImageNet dataset. """

    image_size: ClassVar[Tuple[int, int, int]] = (28, 28, 1)
    n_samples: int = 10_000_000  # some random number just for sake of illustration.
    shuffle: bool = False
    # Path to the imagenet directory.
    path: Path = os.environ.get("IMAGENET_DIR", "data/imagenet")


@dataclass
class ModelConfig:
    """ Configuration options for the Model. """

    # Learning rate.
    lr: float = 3e-4


@dataclass
class SimpleCNNConfig(ModelConfig):
    """ Configuration options for a simple CNN model. """

    lr: float = 1e-3


@dataclass
class ResNetConfig(ModelConfig):
    """ Configuration options for the ResNet model. """

    lr: float = 1e-6


def main():
    parser = ArgumentParser(description=__doc__)
    add_independent_subparsers(
        parser,
        dataset={"mnist": MnistConfig, "imagenet": ImageNetConfig},
        model={"simple_cnn": SimpleCNNConfig, "resnet": ResNetConfig},
    )

    args = parser.parse_args()
    dataset_config: DatasetConfig = args.dataset
    model_config: ModelConfig = args.model

    print(f"Args: {args}")
    print(f"Dataset config: {dataset_config}")
    print(f"Model config: {model_config}")


if __name__ == "__main__":
    main()
