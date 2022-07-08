from pathlib import Path
from typing import Callable, List, Union

from PIL import Image as PILImage
import numpy as np
from napari.layers import Image
from skimage.io import imread

FilePath = Union[Path, str]
ArrayLike = Union[
    np.ndarray, "dask.array.Array"
]  # could add other array types if needed

def flatfield_correct(array:ArrayLike, corr) -> ArrayLike:
    print("hello!")
    return corr(array)

def transpose(array: ArrayLike) -> ArrayLike:
    return array.T

def flip_x(array: ArrayLike) -> ArrayLike:
    return  np.flip(array, axis=1)

def flip_y(array: ArrayLike) -> ArrayLike:
    return  np.flip(array, axis=0)

def crop_black_border(array: ArrayLike, border_width: int = 40) -> ArrayLike:
    """
    Crops away the band of black pixels on the Nikon camera used in our lab.
    """
    right = -border_width if border_width > 0 else None
    return array[border_width:right, border_width:right]


def subsample(array: ArrayLike, factor: int = 4, method="slice") -> ArrayLike:
    """
    Subsamples the input array along all dimensions using the given
    factor. 'slice' method simply slices with factor as the stride,
    which is fast but leads to sampling artefact. Other methods
    should be added (just dispatch to skimage).
    """
    if method == "slice":
        return array[:, ::factor, ::factor]  # todo: support generic nD
    else:
        raise NotImplementedError("only supporting slice for now")


def load_image(
    file: FilePath, transforms: List[Callable[[ArrayLike], ArrayLike]] = None
) -> np.ndarray:
    img = imread(file)
    # if img.ndim == 2:
    #    img = np.expand_dims(img, axis=0)
    if transforms is not None:
        for t in transforms:
            img = t(img)
    return img

def load_image_from_stack(
    file: FilePath, frame: int = 0, transforms: List[Callable[[ArrayLike], ArrayLike]] = None
) -> np.ndarray:
    im = PILImage.open(file)
    im.seek(frame) #move to selected frame
    h,w = np.shape(im)
    img = np.zeros((h,w))
    img = np.array(im).astype(np.uint8) #extract frame
    # if img.ndim == 2:
    #    img = np.expand_dims(img, axis=0)
    if transforms is not None:
        for t in transforms:
            img = t(img)
    return img

