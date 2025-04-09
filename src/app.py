import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
import onnxruntime as ort
from PIL import Image
import io
import cv2
import os
from pathlib import Path
from config import models_dir  # Assuming this is defined in config.py

app = FastAPI(title="Style Transfer API")

# Load ONNX model
onnx_model_path = "style_transfer_mosaic.onnx"
ort_session = ort.InferenceSession(os.path.join(models_dir, onnx_model_path))
input_name = ort_session.get_inputs()[0].name

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to numpy array suitable for ONNX model.
    Matches: ToTensor() + x.mul(255)
    """
    img_array = image.astype(np.float32)[..., ::-1]  # HWC, float32, BGR
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW
    img_array = np.expand_dims(img_array, axis=0)   # BCHW
    return img_array

def postprocess_output(output: np.ndarray) -> np.ndarray:
    """
    Convert model output to image format, matching original pipeline.
    """
    output = output.squeeze(0)  # [C,H,W]
    output = np.transpose(output, (1, 2, 0))  # [H,W,C]
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def transfer_color(source: np.ndarray, target: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """
    Transfers color information from a source image to a target image using the YCrCb color space.
    This is useful for preserving source color characteristics during style transfer.

    Assumes input images are in RGB format and have shape (H, W, C).

    Args:
        source (np.ndarray): Source image in RGB format (provides color information).
        target (np.ndarray): Target image in RGB format (provides luminance/structure).
        weight (float): Style Y-channel weight. Defaults to 1.0

    Returns:
        np.ndarray: Color-transferred image in BGR format.
    """
    source = np.clip(source, 0, 255)
    target = np.clip(target, 0, 255)
    weight = np.clip(weight, 0, 1)

    # Resize target to match source dimensions
    h, w, _ = source.shape
    target_resized = cv2.resize(target, (w, h), interpolation=cv2.INTER_CUBIC)

    # Extract target luminance and source chroma
    target_gray = cv2.cvtColor(target_resized, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    source_ycc = cv2.cvtColor(source, cv2.COLOR_RGB2YCrCb)
    source_ycc[..., 0] = weight * target_gray + (1 - weight) * source_gray

    # Convert back to BGR
    result = cv2.cvtColor(source_ycc, cv2.COLOR_YCrCb2BGR)
    return np.clip(result, 0, 255)

@app.post("/style-transfer/")
async def style_transfer(
        file: UploadFile = File(...), 
        style_option: str = Form(""),
        keep_colors: bool = Form(False)
    ):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        print(f"Style: {style_option}")
        print(f"Color: {keep_colors}, {type(keep_colors)}")
        original_filename = Path(file.filename).stem
        new_filename = f"{original_filename}_processed.png"

        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)

        # Preprocess for model
        input_array = preprocess_image(image_np)

        # Run inference
        ort_inputs = {input_name: input_array}
        output = ort_session.run(None, ort_inputs)[0]

        # Postprocess output
        processed_image = postprocess_output(output)

        # Color transfer
        if keep_colors:
            final_image = transfer_color(image_np, processed_image)
        else:
            final_image = processed_image

        # Convert to PNG bytes
        _, buffer = cv2.imencode(".png", final_image)
        byte_io = io.BytesIO(buffer)

        return StreamingResponse(
            byte_io,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={new_filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image | Code {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)