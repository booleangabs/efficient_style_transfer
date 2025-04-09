import os
from PIL import Image
import io
import requests
import gradio as gr
import numpy as np
import cv2

# Define relative paths to style preview images
STYLE_IMAGES = {
    "Mosaic": os.path.join(os.path.dirname(__file__), os.pardir, "assets", "style_images", "mosaic.jpg"),
    "The Scream": os.path.join(os.path.dirname(__file__), os.pardir, "assets", "style_images", "Edvard Munch - The Scream.jpg"),
    "Old Canal Port": os.path.join(os.path.dirname(__file__), os.pardir, "assets", "style_images", "Oscar Florianus Bluemner - Old Canal Port.jpg"),
    "Starry Night": os.path.join(os.path.dirname(__file__), os.pardir, "assets", "style_images", "Vincent Van Gogh - Starry Night.jpg")
}

def request_style_transfer(image, style_option, keep_colors):
    if image is None:
        return "Error: No image attached."

    url = "http://127.0.0.1:8000/style-transfer/"
    
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    files = {
        "file": ("image.png", image_bytes, "image/png")
    }

    data = {
        "style_option": style_option,
        "keep_colors": str(keep_colors).lower()
    }

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result_image = Image.open(io.BytesIO(response.content))
        return result_image, result_image.copy()  # Return both final and original for post-processing
    else:
        return f"Erro: {response.status_code} - {response.text}", None

def load_style_image(style_option):
    style_path = STYLE_IMAGES.get(style_option)
    if style_path and os.path.exists(style_path):
        return Image.open(style_path)
    return None

def gamma_transform(image, gamma):
    if image is None:
        return None

    gamma = max(gamma, 0.01)
    inv_gamma = 1.0 / gamma
    img_array = np.array(image).astype(np.float32) / 255.0
    corrected = np.power(img_array, inv_gamma)
    corrected = (corrected * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(corrected)

def sharpen_image(image, apply_mask, radius=3, amount=1.5):
    if image is None or not apply_mask:
        return image

    img_array = np.array(image).astype(np.float32)

    blurred = cv2.GaussianBlur(img_array, (0, 0), radius)

    # Unsharp mask: original + amount * (original - blurred)
    sharpened = cv2.addWeighted(img_array, 1 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return Image.fromarray(sharpened)

def apply_bilateral_filter(image, diameter):
    if image is None or diameter <= 1:
        return image

    img_array = np.array(image)
    filtered = cv2.bilateralFilter(img_array, d=diameter, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(filtered)

def postprocess_image(original, gamma, sharpen, bilateral_d):
    if original is None:
        return None
    img = gamma_transform(original, gamma)
    img = sharpen_image(img, sharpen)
    img = apply_bilateral_filter(img, bilateral_d)
    return img

with gr.Blocks() as frontend:
    gr.Markdown("## 🎨 **Style Transfer**")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Input image", height=616)
        with gr.Column():
            default_style = None#"Mosaic"
            style_dropdown = gr.Dropdown(
                choices=list(STYLE_IMAGES.keys()),
                value=default_style,
                label="Select style"
            )
            style_img_default = load_style_image(default_style)
            style_img = gr.Image(label="Style preview", interactive=False, height=512)#, value=style_img_default)

    transfer_color = gr.Checkbox(
        label="Keep original colors",
        value=True
    )

    output_img = gr.Image(label="Output image", height=512)
    original_output_img = gr.State()  # Store unmodified result
    transfer_btn = gr.Button("Run Style Transfer")

    style_dropdown.change(
        load_style_image,
        inputs=[style_dropdown],
        outputs=[style_img]
    )

    transfer_btn.click(
        request_style_transfer,
        inputs=[input_img, style_dropdown, transfer_color],
        outputs=[output_img, original_output_img]
    )

    gr.Markdown("### 🔧 Post-processing")
    gamma_slider = gr.Slider(0.1, 2.0, value=1.0, step=0.01, label="Gamma Correction")
    sharpen_checkbox = gr.Checkbox(label="Apply Sharpening", value=False)
    bilateral_slider = gr.Slider(0, 25, value=0, step=1, label="Bilateral Filter Diameter")


    # Update output based on original image and postprocessing settings
    gamma_slider.change(
        postprocess_image,
        inputs=[original_output_img, gamma_slider, sharpen_checkbox, bilateral_slider],
        outputs=[output_img]
    )

    sharpen_checkbox.change(
        postprocess_image,
        inputs=[original_output_img, gamma_slider, sharpen_checkbox, bilateral_slider],
        outputs=[output_img]
    )

    bilateral_slider.change(
        postprocess_image,
        inputs=[original_output_img, gamma_slider, sharpen_checkbox, bilateral_slider],
        outputs=[output_img]
    )

frontend.launch()
