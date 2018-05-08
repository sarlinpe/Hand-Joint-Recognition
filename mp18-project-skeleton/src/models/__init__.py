"""Model definitions (one class per file) to define NN architectures."""
from .example import ExampleNet
from .resnet50 import Resnet50

__all__ = ('ExampleNet', 'Resnet50')
