import csv
import random

def gerar_amostras(nome_cor, h_range, s_range, v_range, quantidade=1000):
    amostras = []
    for _ in range(quantidade):
        h = random.randint(h_range[0], h_range[1])
        s = random.randint(s_range[0], s_range[1])
        v = random.randint(v_range[0], v_range[1])
        amostras.append([h, s, v, nome_cor])
    return amostras

print("Iniciando a criação do banco de cores do DaltonSpectra...")

dados_completos = []

# --- 1. AS CORES BÁSICAS E MODIFICADORES ---
# O formato do OpenCV para HSV é: H (0-179), S (0-255), V (0-255)

# Vermelho é especial no OpenCV, pois fica no começo (0-10) e no final (170-179) do cilindro
dados_completos.extend(gerar_amostras("Vermelho", (0, 10), (150, 255), (120, 255)))
dados_completos.extend(gerar_amostras("Vermelho", (170, 179), (150, 255), (120, 255)))
dados_completos.extend(gerar_amostras("Vermelho Escuro", (0, 10), (150, 255), (50, 119)))
dados_completos.extend(gerar_amostras("Vermelho Escuro", (170, 179), (150, 255), (50, 119)))

# Laranja e Amarelo
dados_completos.extend(gerar_amostras("Laranja", (11, 22), (150, 255), (150, 255)))
dados_completos.extend(gerar_amostras("Amarelo", (23, 35), (100, 255), (150, 255)))

# Verdes
dados_completos.extend(gerar_amostras("Verde Claro", (36, 85), (80, 200), (200, 255)))
dados_completos.extend(gerar_amostras("Verde", (36, 85), (150, 255), (120, 199)))
dados_completos.extend(gerar_amostras("Verde Escuro", (36, 85), (150, 255), (40, 119)))

# Azuis
dados_completos.extend(gerar_amostras("Azul Claro", (90, 130), (80, 200), (200, 255)))
dados_completos.extend(gerar_amostras("Azul", (90, 130), (150, 255), (120, 199)))
dados_completos.extend(gerar_amostras("Azul Escuro", (90, 130), (150, 255), (40, 119)))

# Roxo, Rosa e Marrom
dados_completos.extend(gerar_amostras("Roxo", (131, 155), (100, 255), (100, 255)))
dados_completos.extend(gerar_amostras("Rosa", (156, 169), (100, 255), (150, 255)))
dados_completos.extend(gerar_amostras("Marrom", (10, 25), (100, 255), (30, 100)))

# --- 2. OS NEUTROS (Luz e Sombra definem essas cores, não o Matiz) ---
# Preto: O brilho (V) é quase zero, independente da cor (H).
dados_completos.extend(gerar_amostras("Preto", (0, 179), (0, 255), (0, 45)))

# Branco: O brilho (V) é alto e a saturação (S) é quase zero.
dados_completos.extend(gerar_amostras("Branco", (0, 179), (0, 35), (200, 255)))

# Cinza: Saturação baixa, mas brilho mediano.
dados_completos.extend(gerar_amostras("Cinza", (0, 179), (0, 45), (46, 199)))

# --- 3. SALVANDO O ARQUIVO ---
# Embaralhamos os dados para o algoritmo de IA não aprender de forma "viciada"
random.shuffle(dados_completos)

with open('banco_de_cores.csv', mode='w', newline='', encoding='utf-8') as arquivo:
    writer = csv.writer(arquivo)
    writer.writerow(['H', 'S', 'V', 'Cor']) # Cabeçalho
    writer.writerows(dados_completos)

print(f"Sucesso! Arquivo 'banco_de_cores.csv' criado com {len(dados_completos)} exemplos simulados.")