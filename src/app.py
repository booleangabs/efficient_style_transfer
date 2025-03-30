import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
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
onnx_model_path = "edge_detector.onnx"
ort_session = ort.InferenceSession(os.path.join(models_dir, onnx_model_path))
input_name = ort_session.get_inputs()[0].name

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL image to numpy array suitable for ONNX model.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW
    img_array = np.expand_dims(img_array, axis=0)   # BCHW
    
    return img_array

def postprocess_output(output: np.ndarray) -> np.ndarray:
    """
    Convert model output to image format.
    """
    output = output.squeeze(0)  # Remove batch dimension
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = (output * 255).clip(0, 255).astype(np.uint8)
    output = output.squeeze(-1)  # Remove channel dimension if 1
    
    return output

@app.post("/style-transfer/")
async def style_transfer(file: UploadFile = File(...)):
    """
    Endpoint to detect edges in an uploaded image.
    Returns the edge-detected image as a PNG.
    """
    try:
        # Check file
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
    
        # Get name without extension
        original_filename = Path(file.filename).stem
        new_filename = f"{original_filename}_processed.png"

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess for model
        input_array = preprocess_image(image)
        
        # Run inference
        ort_inputs = {input_name: input_array}
        output = ort_session.run(None, ort_inputs)[0]
        
        # Postprocess output
        edge_image = postprocess_output(output)
        
        # Convert to PNG bytes
        _, buffer = cv2.imencode(".png", edge_image)
        byte_io = io.BytesIO(buffer)
        
        # Return as streaming response
        return StreamingResponse(
            byte_io,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={new_filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image | Code {str(e)} | ")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)