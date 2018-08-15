# flake8: noqa

from ._features import ImageDescriptor
from .colourhistogram import ColourHistogram
from .imagegmm import ImageGMM

DESCRIPTORS = {
    ColourHistogram.ftype() : ColourHistogram,
    ImageGMM.ftype() : ImageGMM
}