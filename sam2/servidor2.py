import cv2
import numpy as np
import torch
import base64
import pandas as pd # NOVA: Para ler o CSV
from sklearn.neighbors import KNeighborsClassifier # NOVA: A IA classificadora
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse  
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. CONFIGURAÇÃO DO SAM 2 ---
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print(f"SAM 2 Pronto usando: {device}!")

# --- 2. TREINAMENTO DA IA DE CORES (Roda apenas uma vez ao ligar) ---
print("Treinando o modelo de cores...")
df_cores = pd.read_csv("banco_de_cores.csv")
X = df_cores[['H', 'S', 'V']].values # As características
y = df_cores['Cor'].values           # Os nomes das cores

# Cria a IA que vai olhar para os 5 "vizinhos" mais próximos no cilindro 3D
modelo_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
modelo_knn.fit(X, y)
print("Modelo de cores pronto e treinado!")

# --- 3. FUNÇÕES AUXILIARES ---
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
    Usa Machine Learning (KNN) para prever a cor baseada no nosso CSV de treinamento.
    """
    pixel_rgb = np.uint8([[[rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]]]])
    
    # Converte RGB para HSV
    hsv = cv2.cvtColor(pixel_rgb, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Pede para a IA prever o nome da cor!
    caracteristicas_hsv = np.array([[h, s, v]])
    cor_predita = modelo_knn.predict(caracteristicas_hsv)[0]
    
    return cor_predita

# --- 4. CONFIGURAÇÃO DA API FASTAPI ---
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