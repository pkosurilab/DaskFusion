from pathlib import Path
from typing import Callable, List, Union
import dask.array
from PIL import Image as PILImage
import numpy as np
from skimage.io import imread

FilePath = Union[Path, str]
ArrayLike = Union[
    np.ndarray, "dask.array.Array"
]  # could add other array types if needed

def flatfield_correct(array:ArrayLike, corr) -> ArrayLike:
    #print("we are correcting", corr)
    if corr == False:
        minVal = 0
        maxVal = 65535
        im = np.clip(array,minVal,maxVal)
        im2 = ((im-minVal)/(maxVal-minVal)) * 255
        im3 = im2.astype(np.uint8)
        return im3
    else:
        im = corr(array)
        im2 = np.array(im)
        #print(np.min(im2), np.max(im2))
        minVal= 4000.0
        maxVal = 120000.0
        im8 = np.clip(im2, minVal, maxVal)
        im9 = ((im8-minVal)/(maxVal-minVal)) * 255
        im10 = im9.astype(np.uint8)
        return im10

def transpose(array: ArrayLike) -> ArrayLike:
    return array.T

def flip_x(array: ArrayLike) -> ArrayLike:
    return  np.flip(array, axis=1)

def flip_y(array: ArrayLike) -> ArrayLike:
    return  np.flip(array, axis=0)

def crop_black_border(array: ArrayLike, border_width: int = 80) -> ArrayLike:
    """
    Crops away the band of black pixels on the Nikon camera used in our lab.
    """
    
    right = -border_width
    #print(array.shape)
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
    file: FilePath, frame: int = 0, transforms: List[Callable[[ArrayLike], ArrayLike]] = None, corr = None
) -> np.ndarray:
    im = PILImage.open(file)
    im.seek(frame) #move to selected frame
    h,w = np.shape(im)
    img = np.zeros((h,w))
    img = np.array(im) #extract frame
    # if img.ndim == 2:
    #    img = np.expand_dims(img, axis=0)
    if transforms is not None:
        for t in transforms:
            try: 
                if t == flatfield_correct:
                    img = t(img, corr)
                else:
                    img = t(img)
            except AttributeError:
                print(file)
            
    return img

def dead_pixels(array: ArrayLike):
    #print(array.shape)
    m=np.max(array)
    deadBool = (array==m)
    deadPixels = np.argwhere(deadBool == True)
    for x, y in deadPixels:
        n=array[x-1:x+2, y-1:y+2]
        sumN = np.sum(n) - m
        avgN = sumN / 8
        array[x,y] = avgN.astype(int)
    return array