"""JordanHeritageVision Gradio demo. Run from the project root: python app.py"""
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps

from heritagevision.checkpoints import load_checkpoint

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "EfficientNetB0": ("efficientnet_b0", "outputs/experiment_02/checkpoints/best_model.pt"),
    "ResNet101": ("resnet101", "outputs/experiment_04_resnet101/checkpoints/best_model.pt"),
    "VGG16": ("vgg16", "outputs/experiment_06_vgg16/checkpoints/best_model.pt"),
    "InceptionV3": ("inception_v3", "outputs/experiment_08_inceptionv3/checkpoints/best_model.pt"),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIO_MAJOR_VERSION = int(gr.__version__.split(".")[0])

CSS = """
.gradio-container {max-width: 1080px !important; margin: auto !important;}
#hero {text-align: center; padding: 18px 0 5px 0;}
#hero h1 {font-size: 2.3rem; font-weight: 800; margin-bottom: 6px;}
#hero p {color: var(--body-text-color-subdued);}
#upload-zone, #result-image {border-radius: 15px; overflow: hidden;}
#run-button {font-weight: 700; font-size: 1.06rem;}
#results-heading {padding: 6px 0;}
"""


@lru_cache(maxsize=1)
def get_model(selection: str):
    """Keep at most one large model resident in memory."""
    model_name, relative_path = CHECKPOINTS[selection]
    path = ROOT / relative_path
    if not path.is_file():
        raise gr.Error(
            f"Missing checkpoint for {selection}: {relative_path}. "
            "Copy your trained best_model.pt into the expected folder."
        )

    # load_checkpoint uses weights_only=True; load only checkpoints you own/trust.
    model, ckpt, transform = load_checkpoint(path, DEVICE)
    if ckpt.get("model_name") != model_name:
        raise gr.Error(
            f"Model mismatch: {selection} points to checkpoint for "
            f"{ckpt.get('model_name')!r}."
        )
    classes = ckpt.get("classes", [])
    if len(classes) != 6 or len(set(classes)) != 6:
        raise gr.Error("Checkpoint must define six unique class labels.")
    return model, list(classes), transform


def classify(image: Image.Image | None, selection: str):
    if image is None:
        raise gr.Error("Upload an image first, then click Classify.")
    if selection not in CHECKPOINTS:
        raise gr.Error("Choose a model before classifying.")

    model, classes, transform = get_model(selection)
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        if not isinstance(logits, torch.Tensor):
            # InceptionV3 returns a Tensor when eval mode is enabled with AuxLogits disabled.
            raise gr.Error("Expected a single classification Tensor from the selected model.")
        scores = torch.softmax(logits, dim=1)[0].cpu().tolist()

    ranked = sorted(zip(classes, scores), key=lambda pair: pair[1], reverse=True)
    predicted, confidence = ranked[0]
    table = "| Landmark | Softmax score |\n|:--|--:|\n" + "\n".join(
        f"| {name} | {score * 100:.2f}% |" for name, score in ranked
    )
    title = f"### Predicted class: **{predicted}**"
    info = (
        f"**Model:** {selection} &nbsp;·&nbsp; **Top score:** {confidence * 100:.2f}%  "
        "\n\n*Softmax scores are not calibrated probabilities. "
        "This demo is experimental; predictions may be incorrect.*"
    )
    return rgb, title, table, info


def clear_results():
    return None, None, "### Predicted class: —", "", ""


def build_demo():
    # Gradio 6 moved theme/css from Blocks() to launch(). Support both 5 and 6.
    blocks_options = {"title": "JordanHeritageVision | Landmark Classifier"}
    if GRADIO_MAJOR_VERSION < 6:
        blocks_options.update(theme=gr.themes.Soft(primary_hue="teal"), css=CSS)
    with gr.Blocks(**blocks_options) as demo:
        gr.Markdown(
            "# JordanHeritageVision 🏛️\n"
            "Upload a photo of a Jordanian landmark, choose one of our trained "
            "CNN models, and see its prediction across six classes.",
            elem_id="hero",
        )
        with gr.Column():
            image_input = gr.Image(
                label="1 · Drag an image here or click to upload",
                type="pil", image_mode="RGB", sources=["upload"],
                height=320, elem_id="upload-zone",
            )
            selection = gr.Radio(
                choices=list(CHECKPOINTS), value="EfficientNetB0",
                label="2 · Select a trained model",
            )
            with gr.Row():
                classify_btn = gr.Button("Classify image", variant="primary", elem_id="run-button")
                reset_btn = gr.Button("Clear", variant="secondary")
        gr.Markdown("---\n## Classification result")
        result_title = gr.Markdown("### Predicted class: —", elem_id="results-heading")
        with gr.Row():
            result_image = gr.Image(label="Uploaded image", interactive=False,
                                    height=320, elem_id="result-image")
            with gr.Column():
                gr.Markdown("### Scores for all six landmarks")
                probabilities = gr.Markdown("Upload an image and select **Classify image**.")
                model_info = gr.Markdown("")
        gr.Markdown(
            "*The four selectable models were trained for this project. "
            "Scores may be overconfident; a prediction is not a verified location.*"
        )
        classify_btn.click(
            fn=classify, inputs=[image_input, selection],
            outputs=[result_image, result_title, probabilities, model_info],
            concurrency_limit=1,
        )
        reset_btn.click(
            fn=clear_results, inputs=[],
            outputs=[image_input, result_image, result_title, probabilities, model_info],
        )
    return demo


if __name__ == "__main__":
    launch_options = {"inbrowser": True, "share": True}
    if GRADIO_MAJOR_VERSION >= 6:
        launch_options.update(theme=gr.themes.Soft(primary_hue="teal"), css=CSS)
    build_demo().launch(**launch_options)
