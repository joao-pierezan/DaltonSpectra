import cv2
import numpy as np
import torch
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# 1. Configurações Iniciais do SAM 2
# ==========================================
print("Carregando modelo SAM 2 na memória... (Isso pode levar alguns segundos)")
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print(f"SAM 2 Pronto usando: {device}!")

# ==========================================
# 2. Funções de IA de Cores
# ==========================================
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
    # Converte a cor RGB (que veio do K-Means) para o formato do OpenCV
    pixel_rgb = np.uint8([[[rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]]]])
    
    # Transforma de RGB para HSV (A mágica acontece aqui)
    hsv = cv2.cvtColor(pixel_rgb, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 1. Filtros de Luz e Sombra extremos (Escala de Cinza)
    if v < 40: return "Preto"
    
    # Aumentamos de 45 para 50 para perdoar um pouco mais de cor na luz
    if s < 50: 
        if v > 200: return "Branco"
        elif v > 160: return "Branco Gelo / Cinza Claro"
        else: return "Cinza Escuro" if v < 100 else "Cinza"

    # 2. Identificação da Cor Real pela Matiz (Roda de Cores)
    # No OpenCV o H (Matiz) vai de 0 a 179. S e V vão de 0 a 255.
    
    if (h < 10) or (h >= 170): 
        if s < 150 and v > 150: return "Salmão / Rosa Chá"
        return "Vinho / Vermelho Escuro" if v < 120 else "Vermelho"
        
    elif 10 <= h < 22: 
        # Área dos laranjas, marrons e beges
        if v > 180 and s < 130: return "Bege"
        if v < 150: return "Marrom Escuro" if v < 100 else "Marrom"
        if s < 180 and v > 180: return "Coral"
        return "Laranja Escuro" if v < 200 else "Laranja"
        
    elif 22 <= h < 35: 
        # Área dos amarelos
        if v > 180 and s < 120: return "Bege Amarelado / Areia"
        if v < 180: return "Mostarda / Ouro"
        return "Amarelo"
        
    elif 35 <= h < 85: 
        # Área dos verdes
        if h > 70 and s < 100 and v > 150: return "Verde Água"
        if v < 100: return "Verde Escuro / Musgo"
        if h < 45: return "Verde Limão"
        return "Verde"
        
    elif 85 <= h < 100: 
        # Ciano / Turquesa
        if v > 180 and s < 150: return "Azul Bebê"
        return "Turquesa" if s > 150 else "Ciano"
        
    elif 100 <= h < 135: 
        # Azuis
        # ARMADILHA: Se a saturação for baixa (mesmo passando de 50), é só a luz fria enganando a câmera!
        if s < 75: return "Cinza / Cinza Frio" 
        
        if v < 100: return "Azul Marinho"
        if s < 120 and v > 180: return "Azul Claro"
        return "Azul"
        
    elif 135 <= h < 155: 
        # Roxos / Lilás
        if v > 180 and s < 120: return "Lavanda / Lilás"
        return "Roxo Escuro" if v < 120 else "Roxo"
        
    elif 155 <= h < 170: 
        # Rosas
        if v < 150: return "Rosa Escuro / Magenta"
        if s > 180: return "Pink / Rosa Choque"
        return "Rosa Claro"

    return "Cor Indefinida"

# ==========================================
# 3. Servidor Web (FastAPI)
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analisar")
async def analisar_clique(x: int = Form(...), y: int = Form(...), arquivo: UploadFile = File(...)):
    # 1. Lê a imagem
    conteudo = await arquivo.read()
    np_arr = np.frombuffer(conteudo, np.uint8)
    imagem_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 2. Roda o SAM 2
    imagem_rgb = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2RGB)
    predictor.set_image(imagem_rgb)
    ponto = np.array([[x, y]])
    label = np.array([1]) 
    masks, _, _ = predictor.predict(point_coords=ponto, point_labels=label, multimask_output=False)
    mascara = np.squeeze(masks[0]).astype(bool)
    mascara_uint8 = (mascara * 255).astype(np.uint8)

    # 3. Calcula a Cor
    cor_dominante_bgr = obter_cor_dominante(imagem_cv2, mascara_uint8)
    nome_cor = obter_nome_cor((cor_dominante_bgr[2], cor_dominante_bgr[1], cor_dominante_bgr[0]))

    # 4. CRIA A CAMADA 2: O Objeto com Fundo Transparente (RGBA)
    imagem_rgba = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2BGRA)
    imagem_rgba[mascara_uint8 == 0] = [0, 0, 0, 0] 
    
    _, buffer = cv2.imencode('.png', imagem_rgba)
    imagem_base64 = base64.b64encode(buffer).decode('utf-8')

    # 5. CRIA A CAMADA 3: O Caminho do Contorno (Com trava de segurança!)
    contornos, _ = cv2.findContours(mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caminho_svg = ""
    if contornos: # Só faz o desenho se a IA achou um objeto
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

# LIGA O SERVIDOR PARA O SITE CONSEGUIR CONVERSAR COM ELE
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)