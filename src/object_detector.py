"""
Object Detection - ObjectDetector class
Supports YOLOv5 (PyTorch) and SSD MobileNet (OpenCV)
"""
import time
from pathlib import Path
from typing import Union, List

COCO_LABELS = [
    "background","person","bicycle","car","motorcycle","airplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
    "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

class ObjectDetector:
    def __init__(self, backend='yolo', confidence_threshold=0.5):
        if backend not in ('yolo','opencv'):
            raise ValueError("backend must be 'yolo' or 'opencv'")
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        if self.backend == 'yolo':   self._load_yolo()
        else:                        self._load_opencv()

    def _load_yolo(self):
        try:
            import torch
            print("Loading YOLOv5s...")
            self._model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
            self._model.conf = self.confidence_threshold
            print(f"[YOLO] YOLOv5s loaded")
        except ImportError:
            raise ImportError("Run: pip install torch torchvision")

    def _load_opencv(self):
        try:
            import cv2, urllib.request
            md  = Path(__file__).parent.parent / 'models'
            md.mkdir(exist_ok=True)
            cfg = md / 'ssd_mobilenet_v3.pbtxt'
            pb  = md / 'frozen_inference_graph.pb'
            base= "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/"
            if not cfg.exists():
                urllib.request.urlretrieve(
                    base + "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt", cfg)
            self._model = cv2.dnn_DetectionModel(str(pb), str(cfg))
            self._model.setInputSize(320,320)
            self._model.setInputScale(1/127.5)
            self._model.setInputMean((127.5,127.5,127.5))
            self._model.setInputSwapRB(True)
            self._cv2 = cv2
            print("[OpenCV] SSD MobileNet loaded")
        except ImportError:
            raise ImportError("Run: pip install opencv-python")

    def detect(self, image_source) -> List[dict]:
        t0 = time.time()
        if self.backend == 'yolo':
            dets = self._detect_yolo(image_source)
        else:
            dets = self._detect_opencv(image_source)
        ms = round((time.time()-t0)*1000,1)
        for d in dets: d['inference_ms'] = ms
        return dets

    def _detect_yolo(self, src):
        res  = self._model(str(src))
        rows = res.pandas().xyxy[0]
        return [{'label':r['name'],
                 'confidence':round(float(r['confidence']),4),
                 'bbox':{'x':int(r['xmin']),'y':int(r['ymin']),
                         'width':int(r['xmax']-r['xmin']),'height':int(r['ymax']-r['ymin'])}}
                for _,r in rows.iterrows()]

    def _detect_opencv(self, src):
        img = self._cv2.imread(str(src))
        if img is None: raise FileNotFoundError(f"Cannot read: {src}")
        ids, confs, boxes = self._model.detect(img, confThreshold=self.confidence_threshold)
        dets = []
        if len(ids) > 0:
            for cid, conf, box in zip(ids.flatten(), confs.flatten(), boxes):
                label = COCO_LABELS[cid] if cid < len(COCO_LABELS) else f"class_{cid}"
                dets.append({'label':label,'confidence':round(float(conf),4),
                             'bbox':{'x':int(box[0]),'y':int(box[1]),
                                     'width':int(box[2]),'height':int(box[3])}})
        return dets

    def annotate_image(self, image_path, output_path):
        import cv2
        img  = cv2.imread(str(image_path))
        dets = self.detect(image_path)
        COLORS = [(46,232,160),(232,180,46),(46,141,232),(232,77,46)]
        for i,d in enumerate(dets):
            b = d['bbox']
            c = COLORS[i % len(COLORS)]
            cv2.rectangle(img,(b['x'],b['y']),(b['x']+b['width'],b['y']+b['height']),c,2)
            cv2.putText(img,f"{d['label']} {d['confidence']:.0%}",
                       (b['x'],max(b['y']-6,14)),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,2)
        cv2.imwrite(str(output_path), img)
        print(f"Saved -> {output_path}")
