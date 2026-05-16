import cv2
import numpy as np
import torch
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse  
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print(f"SAM 2 Pronto usando: {device}!")


def obter_cor_dominante(imagem, mascara_uint8, k=3):
    pixels = imagem[mascara_uint8 > 0]
    if len(pixels) == 0: return (0, 0, 0)
    pixels = np.float32(pixels)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, rotulos, centros = cv2.kmeans(pixels, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, contagens = np.unique(rotulos, return_counts=True)
    indice_dominante = np.argmax(contagens)
    cor_bgr = centros[indice_dominante]
    return (int(cor_bgr[0]), int(cor_bgr[1]), int(cor_bgr[2]))

def obter_nome_cor(rgb_tuple):
    """
    Traduz a cor usando como o olho humano enxerga (HSV), resolvendo 
    problemas de sombras (azul virando preto) e tons (laranja virando vermelho).
    """
    pixel_rgb = np.uint8([[[rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]]]])
    
    hsv = cv2.cvtColor(pixel_rgb, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 40: return "Preto"

    if s < 50: 
        if v > 200: return "Branco"
        elif v > 160: return "Branco / Cinza Claro"
        else: return "Cinza Escuro" if v < 100 else "Cinza"

    
    if (h < 10) or (h >= 170): 
        if s < 150 and v > 150: return "Salmão"
        return "Vinho / Vermelho Escuro" if v < 120 else "Vermelho"
        
    elif 10 <= h < 22: 
        
        if v > 180 and s < 130: return "Bege"
        if v < 150: return "Marrom Escuro" if v < 100 else "Marrom"
        if s < 180 and v > 180: return "Coral"
        return "Laranja Escuro" if v < 200 else "Laranja"
        
    elif 22 <= h < 35: 
        if v > 180 and s < 120: return "Amarelo Claro"
        if v < 180: return "Amarelo Escuro/ Queimado"
        return "Amarelo"
        
    elif 35 <= h < 85: 
        if h > 70 and s < 100 and v > 150: return "Verde Água"
        if v < 100: return "Verde Escuro"
        if h < 45: return "Verde Neon"
        return "Verde"
        
    elif 85 <= h < 100: 
        if v > 180 and s < 150: return "Azul Bebê"
        return "Turquesa" if s > 150 else "Ciano"
        
    elif 100 <= h < 135: 
        if s < 75: return "Cinza" 
        
        if v < 100: return "Azul Marinho"
        if s < 120 and v > 180: return "Azul Claro"
        return "Azul"
        
    elif 135 <= h < 155: 
        if v > 180 and s < 120: return "Lilás"
        return "Roxo Escuro" if v < 120 else "Roxo"
        
    elif 155 <= h < 170: 

        if v < 150: return "Rosa Escuro"
        if s > 180: return "Rosa Choque"
        return "Rosa Claro"

    return "Cor Indefinida"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def pagina_principal():
    return FileResponse("showpage.html")

@app.get("/app")
async def pagina_ferramenta():
    return FileResponse("app.html")

@app.post("/analisar")
async def analisar_clique(x: int = Form(...), y: int = Form(...), arquivo: UploadFile = File(...)):
    conteudo = await arquivo.read()
    np_arr = np.frombuffer(conteudo, np.uint8)
    imagem_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    imagem_rgb = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2RGB)
    predictor.set_image(imagem_rgb)
    ponto = np.array([[x, y]])
    label = np.array([1]) 
    masks, _, _ = predictor.predict(point_coords=ponto, point_labels=label, multimask_output=False)
    mascara = np.squeeze(masks[0]).astype(bool)
    mascara_uint8 = (mascara * 255).astype(np.uint8)

    cor_dominante_bgr = obter_cor_dominante(imagem_cv2, mascara_uint8)
    nome_cor = obter_nome_cor((cor_dominante_bgr[2], cor_dominante_bgr[1], cor_dominante_bgr[0]))

    imagem_rgba = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2BGRA)
    imagem_rgba[mascara_uint8 == 0] = [0, 0, 0, 0] 
    
    _, buffer = cv2.imencode('.png', imagem_rgba)
    imagem_base64 = base64.b64encode(buffer).decode('utf-8')

    contornos, _ = cv2.findContours(mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caminho_svg = ""
    if contornos: 
        maior_contorno = max(contornos, key=cv2.contourArea) 
        for i, ponto_contorno in enumerate(maior_contorno):
            px, py = ponto_contorno[0]
            if i == 0:
                caminho_svg += f"M {px} {py} "
            else:
                caminho_svg += f"L {px} {py} "
        caminho_svg += "Z"

    return {
        "status": "sucesso",
        "cor_predominante": nome_cor,
        "imagem_resultado": imagem_base64, 
        "caminho_svg": caminho_svg         
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)