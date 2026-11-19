"""Headless image prediction: python -m app.cli predict IMAGE --output FILE."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Mining detector command-line tools')
    commands = parser.add_subparsers(dest='command', required=True)
    predict = commands.add_parser('predict', help='Analyze one image using the active model')
    predict.add_argument('image', type=Path)
    predict.add_argument('--output', type=Path, help='Save prediction JSON')
    predict.add_argument('--heatmap', type=Path, help='Save a Grad-CAM overlay image')
    predict.add_argument('--checkpoint', type=Path, help='Use another checkpoint')
    predict.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    predict.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image not found: {args.image}')
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    from app.core.predictor import Predictor
    torch.set_num_threads(args.threads)
    try:
        predictor = Predictor(checkpoint_path=args.checkpoint,
                              device=None if args.device == 'auto' else args.device)
        result = predictor.predict(args.image, generate_cam=args.heatmap is not None)
        if args.heatmap:
            if result.heatmap_pil is None:
                raise RuntimeError('Grad-CAM generation failed; see logs/app.log')
            args.heatmap.parent.mkdir(parents=True, exist_ok=True)
            result.heatmap_pil.save(args.heatmap)
        payload = json.dumps(result.to_dict(), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + '\n', encoding='utf-8')
        print(payload)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Prediction failed: {exc}\n')


if __name__ == '__main__':
    main()
