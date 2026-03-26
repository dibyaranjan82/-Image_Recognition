"""
Image Recognition - ImageRecognizer class
Supports TensorFlow (MobileNetV2), PyTorch (ResNet50), OpenCV (GoogLeNet)
"""
import json, time
from pathlib import Path
from typing import Union, List

class ImageRecognizer:
    BACKENDS = ['tensorflow', 'torch', 'opencv']

    def __init__(self, backend='tensorflow', top_k=5):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")
        self.backend = backend
        self.top_k   = top_k
        self._model  = None
        self._load_model()

    def _load_model(self):
        if self.backend == 'tensorflow': self._load_tf()
        elif self.backend == 'torch':    self._load_torch()
        elif self.backend == 'opencv':   self._load_opencv()

    def _load_tf(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
            self._model     = MobileNetV2(weights='imagenet')
            self._preprocess = preprocess_input
            self._decode     = decode_predictions
            print(f"[TF] MobileNetV2 loaded (TF {tf.__version__})")
        except ImportError:
            raise ImportError("Run: pip install tensorflow")

    def _load_torch(self):
        try:
            import torch, torchvision.models as models
            import torchvision.transforms as T
            self._model = models.resnet50(pretrained=True)
            self._model.eval()
            self._transform = T.Compose([
                T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            import urllib.request, json as _j
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            with urllib.request.urlopen(url) as r:
                self._labels = _j.loads(r.read().decode())
            self._torch = torch
            print(f"[Torch] ResNet-50 loaded (PyTorch {torch.__version__})")
        except ImportError:
            raise ImportError("Run: pip install torch torchvision")

    def _load_opencv(self):
        try:
            import cv2, urllib.request, numpy as np
            md = Path(__file__).parent.parent / 'models'
            md.mkdir(exist_ok=True)
            proto  = md / 'bvlc_googlenet.prototxt'
            caffe  = md / 'bvlc_googlenet.caffemodel'
            synset = md / 'synset_words.txt'
            base   = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/"
            if not proto.exists():
                print("Downloading GoogLeNet prototxt...")
                urllib.request.urlretrieve(base + "bvlc_googlenet.prototxt", proto)
            if not synset.exists():
                print("Downloading labels...")
                urllib.request.urlretrieve(base + "synset_words.txt", synset)
            if not caffe.exists():
                print("Downloading GoogLeNet weights (~50MB)...")
                urllib.request.urlretrieve("http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel", caffe)
            self._model = cv2.dnn.readNetFromCaffe(str(proto), str(caffe))
            with open(synset) as f:
                self._labels = [l.strip().split(None,1)[1] for l in f]
            self._cv2 = cv2
            self._np  = np
            print(f"[OpenCV] GoogLeNet loaded")
        except ImportError:
            raise ImportError("Run: pip install opencv-python")

    def _load_pil(self, src):
        from PIL import Image
        import numpy as np
        if isinstance(src, np.ndarray):
            return Image.fromarray(src[...,::-1])
        path = str(src)
        if path.startswith('http'):
            import urllib.request, io
            with urllib.request.urlopen(path) as r:
                return Image.open(io.BytesIO(r.read())).convert('RGB')
        return Image.open(path).convert('RGB')

    def predict(self, source) -> List[dict]:
        t0 = time.time()
        if self.backend == 'tensorflow':
            return self._pred_tf(source, t0)
        elif self.backend == 'torch':
            return self._pred_torch(source, t0)
        elif self.backend == 'opencv':
            return self._pred_opencv(source, t0)

    def _pred_tf(self, src, t0):
        import numpy as np
        from tensorflow.keras.preprocessing import image as kimg
        img  = self._load_pil(src).resize((224,224))
        arr  = kimg.img_to_array(img)
        arr  = np.expand_dims(arr, 0)
        arr  = self._preprocess(arr)
        preds = self._model.predict(arr, verbose=0)
        dec  = self._decode(preds, top=self.top_k)[0]
        ms   = round((time.time()-t0)*1000, 1)
        return [{'rank':i+1,'label':l,'confidence':round(float(c),4),'inference_ms':ms}
                for i,(_,l,c) in enumerate(dec)]

    def _pred_torch(self, src, t0):
        img    = self._load_pil(src)
        tensor = self._transform(img).unsqueeze(0)
        with self._torch.no_grad():
            out = self._model(tensor)
        probs = self._torch.nn.functional.softmax(out[0], dim=0)
        top   = self._torch.topk(probs, self.top_k)
        ms    = round((time.time()-t0)*1000, 1)
        return [{'rank':i+1,'label':self._labels[idx.item()],
                 'confidence':round(val.item(),4),'inference_ms':ms}
                for i,(val,idx) in enumerate(zip(top.values, top.indices))]

    def _pred_opencv(self, src, t0):
        img  = self._cv2.imread(str(src))
        blob = self._cv2.dnn.blobFromImage(img, 1, (224,224), (104,117,123))
        self._model.setInput(blob)
        preds = self._model.forward()[0]
        top   = self._np.argsort(preds)[::-1][:self.top_k]
        ms    = round((time.time()-t0)*1000, 1)
        return [{'rank':i+1,'label':self._labels[idx],
                 'confidence':round(float(preds[idx]),4),'inference_ms':ms}
                for i,idx in enumerate(top)]

    def predict_batch(self, sources, verbose=True):
        results = []
        for i,src in enumerate(sources):
            if verbose: print(f"Processing {i+1}/{len(sources)}: {src}")
            try:
                preds = self.predict(src)
                results.append({'source':str(src),'predictions':preds,'error':None})
            except Exception as e:
                results.append({'source':str(src),'predictions':[],'error':str(e)})
        return results

    def save_results(self, results, path):
        with open(path,'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {path}")
