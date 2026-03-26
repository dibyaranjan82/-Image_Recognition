"""
VisionCore CLI
Usage:
    python main.py classify --image photo.jpg
    python main.py detect   --image photo.jpg --annotate
    python main.py batch    --folder data/    --output results.json
    python main.py info     --image photo.jpg
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def cmd_classify(args):
    from src.image_recognizer import ImageRecognizer
    r = ImageRecognizer(backend=args.backend, top_k=args.top_k)
    preds = r.predict(args.image)
    print(f"\n{'─'*55}")
    print(f"  Image   : {args.image}")
    print(f"  Backend : {args.backend}")
    print(f"{'─'*55}")
    print(f"  {'Rank':<5} {'Label':<30} {'Confidence':>12}  ms")
    print(f"{'─'*55}")
    for p in preds:
        print(f"  #{p['rank']:<4} {p['label']:<30} {p['confidence']:>11.2%}  {p['inference_ms']}")
    print()

def cmd_detect(args):
    from src.object_detector import ObjectDetector
    d = ObjectDetector(backend=args.backend, confidence_threshold=args.confidence)
    dets = d.detect(args.image)
    print(f"\nDetected {len(dets)} objects in {args.image}")
    for det in dets:
        b = det['bbox']
        print(f"  {det['label']:20} {det['confidence']:.2%}  bbox=({b['x']},{b['y']},{b['width']}x{b['height']})")
    if args.annotate:
        out = Path(args.image).stem + '_annotated.jpg'
        d.annotate_image(args.image, out)

def cmd_batch(args):
    from src.image_recognizer import ImageRecognizer
    folder = Path(args.folder)
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
    if not imgs: print(f"No images found in {folder}"); return
    r = ImageRecognizer(backend=args.backend, top_k=args.top_k)
    results = r.predict_batch([str(p) for p in imgs])
    for item in results:
        print(f"\n{item['source']}")
        for p in (item['predictions'] or [])[:3]:
            print(f"  #{p['rank']} {p['label']} ({p['confidence']:.2%})")
    if args.output:
        r.save_results(results, args.output)

def cmd_info(args):
    from utils.preprocessing import get_image_info
    info = get_image_info(args.image)
    print(f"\nImage: {args.image}")
    for k,v in info.items(): print(f"  {k:<12}: {v}")

parser = argparse.ArgumentParser(description='VisionCore CLI')
sub = parser.add_subparsers(dest='command', required=True)

p1 = sub.add_parser('classify')
p1.add_argument('--image',   required=True)
p1.add_argument('--backend', default='tensorflow', choices=['tensorflow','torch','opencv'])
p1.add_argument('--top-k',   dest='top_k', type=int, default=5)

p2 = sub.add_parser('detect')
p2.add_argument('--image',      required=True)
p2.add_argument('--backend',    default='yolo', choices=['yolo','opencv'])
p2.add_argument('--confidence', type=float, default=0.5)
p2.add_argument('--annotate',   action='store_true')

p3 = sub.add_parser('batch')
p3.add_argument('--folder',  required=True)
p3.add_argument('--backend', default='tensorflow', choices=['tensorflow','torch','opencv'])
p3.add_argument('--top-k',   dest='top_k', type=int, default=3)
p3.add_argument('--output',  default='')

p4 = sub.add_parser('info')
p4.add_argument('--image', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    {'classify':cmd_classify,'detect':cmd_detect,'batch':cmd_batch,'info':cmd_info}[args.command](args)
