"""Gradio demo: pneumonia screening on a chest X-ray (research prototype, not for clinical use)."""
import numpy as np
import tensorflow as tf
import gradio as gr
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG = 224
MODEL = tf.keras.models.load_model("pneumonia_resnet50.keras")
LAST_CONV = "conv5_block3_out"
DISCLAIMER = ("Research prototype trained on a public paediatric dataset. "
              "Not a medical device. Not for clinical use.")


def grad_cam(x):
    base = MODEL.get_layer("resnet50")
    gm = tf.keras.models.Model(base.inputs, [base.get_layer(LAST_CONV).output, base.output])
    with tf.GradientTape() as tape:
        conv_out, base_out = gm(x)
        p = MODEL.layers[-1](tf.reduce_mean(base_out, axis=(1, 2)))
        loss = p[:, 0]
    grads = tape.gradient(loss, conv_out)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.maximum(tf.reduce_sum(conv_out[0] * w, axis=-1), 0)
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()


def predict(image):
    if image is None:
        return {}, None
    rgb = np.array(image.convert("RGB").resize((IMG, IMG)), dtype=np.float32)
    x = preprocess_input(rgb.copy())[None]
    p = float(MODEL.predict(x, verbose=0)[0, 0])  # P(PNEUMONIA)
    overlay = None
    try:
        cam = cv2.resize(grad_cam(tf.constant(x)), (IMG, IMG))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
        overlay = np.uint8(0.6 * rgb + 0.4 * heat)
    except Exception:
        pass  # keep the prediction even if the heatmap fails
    return {"PNEUMONIA": p, "NORMAL": 1 - p}, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Chest X-ray"),
    outputs=[gr.Label(label="Model output"), gr.Image(label="Grad-CAM (where the model looked)")],
    title="Pneumonia detection on chest X-ray (ResNet50)",
    description=DISCLAIMER,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
