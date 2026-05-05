import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# 1. Funções de Interface e Cores
# ==========================================
ALTURA_BARRA = 90 

def redimensionar_para_tela(imagem, limite_largura=1280, limite_altura=720):
    h, w = imagem.shape[:2]
    escala = min(limite_largura / w, limite_altura / h)
    if escala < 1:
        return cv2.resize(imagem, (int(w * escala), int(h * escala)))
    return imagem

def obter_cor_dominante(imagem, mascara_uint8, k=3):
    """Usa IA (K-Means) para achar a cor mais frequente, ignorando sombras e reflexos"""
    # Pega apenas os pixels que estão dentro da máscara
    pixels = imagem[mascara_uint8 > 0]
    
    if len(pixels) == 0:
        return (0, 0, 0)
        
    # Converte para o formato que o K-Means exige
    pixels = np.float32(pixels)
    
    # Configura e roda o K-Means agrupando em 'k' cores (por padrão, 3)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, rotulos, centros = cv2.kmeans(pixels, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Conta qual grupo de cor tem mais pixels
    _, contagens = np.unique(rotulos, return_counts=True)
    indice_dominante = np.argmax(contagens)
    
    # Pega a cor exata do grupo vencedor
    cor_bgr = centros[indice_dominante]
    
    return (int(cor_bgr[0]), int(cor_bgr[1]), int(cor_bgr[2]))

def obter_nome_cor(rgb):
    """Dicionário expandido para maior acurácia"""
    cores_conhecidas = {
        "Preto": (0, 0, 0), "Branco": (255, 255, 255), "Cinza Escuro": (64, 64, 64),
        "Cinza": (128, 128, 128), "Cinza Claro": (192, 192, 192),
        "Vermelho": (255, 0, 0), "Vinho": (128, 0, 0), "Rosa": (255, 192, 203), "Rosa Choque": (255, 20, 147),
        "Verde Escuro": (0, 100, 0), "Verde": (0, 255, 0), "Verde Claro": (144, 238, 144),
        "Azul Marinho": (0, 0, 128), "Azul": (0, 0, 255), "Azul Claro": (135, 206, 235), "Ciano": (0, 255, 255),
        "Amarelo": (255, 255, 0), "Mostarda": (218, 165, 32),
        "Laranja": (255, 165, 0), "Coral": (255, 127, 80),
        "Roxo": (128, 0, 128), "Lilás": (221, 160, 221),
        "Marrom": (139, 69, 19), "Bege": (245, 245, 220), "Pele / Pêssego": (255, 218, 185)
    }
    
    menor_distancia = float('inf')
    cor_mais_proxima = "Desconhecida"
    
    for nome, valor in cores_conhecidas.items():
        # Distância Euclidiana entre as cores
        dist = sum((a - b) ** 2 for a, b in zip(rgb, valor))
        if dist < menor_distancia:
            menor_distancia = dist
            cor_mais_proxima = nome
            
    return cor_mais_proxima

def montar_interface(imagem, texto_status):
    h, w = imagem.shape[:2]
    largura_tela = max(w, 500)
    tela = np.zeros((h + ALTURA_BARRA, largura_tela, 3), dtype=np.uint8)
    
    tela[0:ALTURA_BARRA, :] = (30, 30, 30)
    offset_x = (largura_tela - w) // 2
    tela[ALTURA_BARRA:, offset_x:offset_x + w] = imagem
    
    cv2.putText(tela, "SAM 2 | Analise de Objeto (IA de Cores)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(tela, texto_status, (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return tela, offset_x

# ==========================================
# 2. Setup do Modelo
# ==========================================
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# ==========================================
# 3. Carregar Imagem
# ==========================================
caminho_imagem = "camiseta.jpg" # <--- NÃO ESQUEÇA DA SUA IMAGEM
imagem_crua = cv2.imread(caminho_imagem)

if imagem_crua is None:
    print("Erro: Imagem nao encontrada.")
    exit()

imagem_original = redimensionar_para_tela(imagem_crua)
imagem_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
predictor.set_image(imagem_rgb)

imagem_tela, margem_x = montar_interface(imagem_original, "Clique em um objeto para destacar")

# ==========================================
# 4. Evento de Clique
# ==========================================
def clique_mouse(event, x, y, flags, param):
    global imagem_tela

    if event == cv2.EVENT_LBUTTONDOWN:
        if y < ALTURA_BARRA or x < margem_x or x > (margem_x + imagem_original.shape[1]): 
            return
            
        y_real = y - ALTURA_BARRA 
        x_real = x - margem_x
        
        ponto = np.array([[x_real, y_real]])
        label = np.array([1]) 

        masks, _, _ = predictor.predict(point_coords=ponto, point_labels=label, multimask_output=False)
        mascara = np.squeeze(masks[0]).astype(bool)

        # Efeito Blur e Borda
        fundo_borrado = cv2.GaussianBlur(imagem_original, (81, 81), 0)
        resultado = np.where(mascara[:, :, None], imagem_original, fundo_borrado)
        
        mascara_uint8 = (mascara * 255).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(resultado, contornos, -1, (255, 255, 255), 2)

        # NOVA IDENTIFICAÇÃO DA COR DOMINANTE
        cor_dominante_bgr = obter_cor_dominante(imagem_original, mascara_uint8)
        
        # Converte para RGB para achar o nome
        rgb_para_nome = (cor_dominante_bgr[2], cor_dominante_bgr[1], cor_dominante_bgr[0])
        nome_cor = obter_nome_cor(rgb_para_nome)

        imagem_tela, _ = montar_interface(resultado, f"Cor Predominante: {nome_cor}")
        cv2.imshow("SAM 2 - Modo Estudio", imagem_tela)

# ==========================================
# 5. Loop Principal
# ==========================================
cv2.namedWindow("SAM 2 - Modo Estudio")
cv2.setMouseCallback("SAM 2 - Modo Estudio", clique_mouse)

while True:
    cv2.imshow("SAM 2 - Modo Estudio", imagem_tela)
    if cv2.waitKey(1) & 0xFF == 27: break

cv2.destroyAllWindows()