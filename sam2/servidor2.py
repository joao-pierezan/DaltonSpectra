import cv2
import numpy as np
import torch
import base64
import pandas as pd # Para ler o CSV
import csv # NOVA: Para escrever no CSV na rota de correção
from sklearn.neighbors import KNeighborsClassifier # A IA classificadora
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

# Variável global para armazenar o modelo
modelo_knn = None

# --- 2. TREINAMENTO DA IA DE CORES ---
def treinar_modelo():
    """Lê o banco de dados e treina o modelo KNN."""
    global modelo_knn
    print("Treinando o modelo de cores...")
    df_cores = pd.read_csv("banco_de_cores.csv")
    X = df_cores[['H', 'S', 'V']].values # As características
    y = df_cores['Cor'].values           # Os nomes das cores

    # Cria a IA que vai olhar para os 5 "vizinhos" mais próximos no cilindro 3D
    modelo_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    modelo_knn.fit(X, y)
    print("Modelo de cores pronto e treinado!")

# Roda a primeira vez ao ligar o servidor
treinar_modelo()

# --- 3. FUNÇÕES AUXILIARES ---
def obter_cor_dominante(imagem, mascara_uint8, k=3):
    # 1. Pega apenas a região da imagem que o SAM 2 recortou
    imagem_recortada = cv2.bitwise_and(imagem, imagem, mask=mascara_uint8)

    # 2. Converte a região para HSV para analisarmos a luz física
    hsv_recortado = cv2.cvtColor(imagem_recortada, cv2.COLOR_BGR2HSV)

    # 3. CRIA A MÁSCARA ANTI-ILUMINAÇÃO RUIM
    # Ignora pixels onde a Saturação é menor que 40 (reflexos brancos/cinzas)
    # Ignora pixels onde o Valor/Brilho é menor que 40 (sombras pretas escuras)
    limite_inferior = np.array([0, 40, 40])
    limite_superior = np.array([179, 255, 255])
    mascara_sem_luz_ruim = cv2.inRange(hsv_recortado, limite_inferior, limite_superior)

    # 4. Junta a máscara do SAM com a nossa máscara anti-reflexo
    mascara_final = cv2.bitwise_and(mascara_uint8, mascara_sem_luz_ruim)

    # 5. Extrai apenas os pixels que sobreviveram aos dois filtros
    pixels_validos = imagem[mascara_final > 0]

    # TRAVA DE SEGURANÇA: E se o objeto for REALMENTE branco, preto ou cinza?
    # Nesse caso, o nosso filtro anti-reflexo vai apagar o objeto inteiro.
    # Se sobrar menos de 50 pixels, nós desativamos o filtro e usamos a máscara original do SAM.
    if len(pixels_validos) < 50:
        pixels_validos = imagem[mascara_uint8 > 0]
        if len(pixels_validos) == 0: 
            return (0, 0, 0) # Prevenção de erro caso não tenha nenhum pixel

    # 6. Aplica o K-Means APENAS nos pixels altamente coloridos
    pixels = np.float32(pixels_validos)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, rotulos, centros = cv2.kmeans(pixels, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Acha o grupo vencedor
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
    
    # Retornamos também o HSV para o frontend saber qual cor exata ele achou
    # Assim, se o usuário corrigir, o frontend manda esses mesmos números de volta
    return cor_predita, h, s, v

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
    
    # Agora a função retorna o nome E os valores HSV
    nome_cor, h_detectado, s_detectado, v_detectado = obter_nome_cor((cor_dominante_bgr[2], cor_dominante_bgr[1], cor_dominante_bgr[0]))

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
        # Enviamos o HSV detectado para o frontend guardar na manga caso precise corrigir
        "h": h_detectado,
        "s": s_detectado,
        "v": v_detectado,
        "imagem_resultado": imagem_base64, 
        "caminho_svg": caminho_svg        
    }

@app.post("/corrigir")
async def corrigir_cor(h: int = Form(...), s: int = Form(...), v: int = Form(...), cor_correta: str = Form(...)):
    """
    Recebe a correção de uma cor e salva MÚLTIPLAS VEZES para vencer os dados sintéticos.
    Cria uma "nuvem" de pontos ao redor da correção humana para garantir que ela ganhe a votação.
    """
    with open("banco_de_cores.csv", mode="a", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        
        # Cria uma zona de impacto ao redor do HSV clicado
        for variacao_h in [-1, 0, 1]:
            for variacao_s in [-3, 0, 3]:
                for variacao_v in [-3, 0, 3]:
                    
                    # Para cada ponto dessa nuvem, nós salvamos 5 vezes (Peso x5)
                    for _ in range(5): 
                        novo_h = max(0, min(179, h + variacao_h))
                        novo_s = max(0, min(255, s + variacao_s))
                        novo_v = max(0, min(255, v + variacao_v))
                        writer.writerow([novo_h, novo_s, novo_v, cor_correta])
    
    # Faz a IA ler o arquivo atualizado (que agora tem um "peso" humano forte)
    treinar_modelo()
    
    return {"status": "sucesso", "mensagem": f"Cor {cor_correta} salva com força total!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)