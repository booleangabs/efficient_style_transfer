import gradio as gr
import requests
from PIL import Image
import io

def transfer_style(image):
    url = "http://127.0.0.1:8000/style-transfer/"  # Ajuste se necessário
    
    if image is None:
        return "Erro: Nenhuma imagem enviada."
    
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        return f"Erro: {response.status_code} - {response.text}"

#Interface
with gr.Blocks() as demo:
    gr.Markdown("## Transferência de Estilo")
    with gr.Row():
        input_img = gr.Image(type="pil", label="Imagem de Entrada")
        style_img = gr.Image(type="pil", label="Imagem de Estilo (não conectada)")
    
    output_img = gr.Image(label="Imagem Processada")
    transfer_btn = gr.Button("Transferir")
    
    transfer_btn.click(transfer_style, inputs=[input_img], outputs=[output_img])

demo.launch()
