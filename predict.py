"""Predict the top-3 landmarks for any local image."""
import argparse
import torch
from PIL import Image, ImageOps
from heritagevision.checkpoints import load_checkpoint


def predict_image(pil_image, model, checkpoint, transform, device, top_k=3):
    img = ImageOps.exif_transpose(pil_image).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=1)[0]
        values, indices = probabilities.topk(min(top_k, len(checkpoint['classes'])))
    return {checkpoint['classes'][i]: float(prob)
            for prob, i in zip(values.cpu().tolist(), indices.cpu().tolist())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--checkpoint', default='outputs/checkpoints/best_model.pt')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt, transform = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        results = predict_image(image, model, ckpt, transform, device)
    for label, probability in results.items():
        print(f'{label:<25} {probability:.2%}')


if __name__ == '__main__':
    main()
