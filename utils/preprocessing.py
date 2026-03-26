"""Preprocessing utilities for image loading, resizing, augmentation."""
import numpy as np
from pathlib import Path
from typing import Union, Tuple

def load_image(source, color_mode='RGB'):
    if isinstance(source, np.ndarray): return source.copy()
    from PIL import Image
    src = str(source)
    if src.startswith('http'):
        import urllib.request, io
        with urllib.request.urlopen(src) as r:
            return np.array(Image.open(io.BytesIO(r.read())).convert('RGB'))
    img = np.array(Image.open(src).convert('RGB'))
    if color_mode == 'BGR': return img[...,::-1].copy()
    return img

def resize_image(img, size=(224,224), keep_aspect=False):
    from PIL import Image
    pil = Image.fromarray(img)
    if keep_aspect:
        pil.thumbnail(size, Image.LANCZOS)
        out = Image.new('RGB', size, (0,0,0))
        out.paste(pil, ((size[0]-pil.width)//2, (size[1]-pil.height)//2))
        return np.array(out)
    return np.array(pil.resize(size, Image.LANCZOS))

def normalize_image(img, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    arr = img.astype(np.float32) / 255.0
    return (arr - np.array(mean)) / np.array(std)

def augment_image(img, flip_h=False, flip_v=False, rotate=0.0, brightness=1.0, noise=0.0):
    from PIL import Image, ImageEnhance
    pil = Image.fromarray(img)
    if flip_h: pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v: pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate:  pil = pil.rotate(rotate)
    if brightness != 1.0: pil = ImageEnhance.Brightness(pil).enhance(brightness)
    result = np.array(pil)
    if noise > 0:
        result = np.clip(result.astype(np.float32) + np.random.normal(0,noise,result.shape), 0, 255).astype(np.uint8)
    return result

def get_image_info(source):
    img = load_image(source)
    h, w = img.shape[:2]
    return {'width':w,'height':h,'channels':img.shape[2] if img.ndim==3 else 1,
            'dtype':str(img.dtype),'size_kb':round(img.nbytes/1024,1)}
