import csv

def classificar_hsv(h, s, v):
    """
    Regras matemáticas otimizadas para o dia a dia de um daltônico.
    Espaço de Cores HSV no OpenCV: H (0-179), S (0-255), V (0-255).
    """
    
    # 1. ESCALA ACROMÁTICA (Preto, Branco e Cinza)
    if v < 40:
        return "Preto"
    if s < 40 and v > 200:
        return "Branco"
    if s < 50 and 40 <= v <= 200:
        return "Cinza"

    # 2. TONS TERROSOS (Marrom e variações escuras de cores quentes)
    if (h <= 20 or h >= 165) and s > 50 and 40 <= v < 130:
        return "Marrom"
    if 20 < h <= 35 and s > 50 and 40 <= v < 120:
        return "Marrom" # Absorve o Amarelo Escuro/Oliva/Ocre como Marrom

    # 3. ESPECTRO CROMÁTICO (Baseado no Matiz - H)
    
    # Vermelhos e variações
    if h <= 10 or h >= 170:
        if v < 150: return "Vermelho Escuro"
        if s < 150 and v > 180: return "Rosa" # Vermelho claro/desbotado
        return "Vermelho"
        
    # Laranja
    elif 10 < h <= 22:
        if v < 120: return "Marrom" # Laranja muito escuro vira marrom no mundo real
        return "Laranja"
        
    # Amarelo e Bege
    elif 22 < h <= 35:
        if s < 120 and v > 180: return "Bege" 
        return "Amarelo"
        
    # Verdes (Claro, Normal e Escuro)
    elif 35 < h <= 85:
        if v < 120: return "Verde Escuro"
        if s < 150 and v > 180: return "Verde Claro"
        return "Verde"
        
    # Azuis (Claro, Normal e Escuro) - Substitui o antigo "Ciano"
    elif 85 < h <= 130:
        if v < 120: return "Azul Escuro"
        if s < 180 and v > 180: return "Azul Claro" # Cores da faixa ciano entram aqui como Azul Claro
        return "Azul"
        
    # Roxo
    elif 130 < h <= 160:
        return "Roxo"
        
    # Rosa
    elif 160 < h < 170:
        return "Rosa"

    return "Indefinido"


def gerar_banco_limpo():
    nome_arquivo = "banco_de_cores.csv"
    
    # Resolução da malha (Equilíbrio perfeito entre precisão e leveza)
    passo_h = 4  
    passo_s = 15 
    passo_v = 15 
    
    total_linhas = 0
    contagem_cores = {}

    with open(nome_arquivo, mode="w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        # Cabeçalho estruturado para o algoritmo KNN
        writer.writerow(["H", "S", "V", "Cor"])
        
        # Varre todo o cilindro matemático HSV
        for h in range(0, 180, passo_h):
            for s in range(0, 256, passo_s):
                for v in range(0, 256, passo_v):
                    
                    cor = classificar_hsv(h, s, v)
                    
                    if cor != "Indefinido":
                        writer.writerow([h, s, v, cor])
                        total_linhas += 1
                        
                        if cor in contagem_cores:
                            contagem_cores[cor] += 1
                        else:
                            contagem_cores[cor] = 1

    # Relatório de criação no terminal
    print("-" * 50)
    print(f"Sucesso! Novo arquivo '{nome_arquivo}' criado.")
    print(f"Total de registros matemáticos balanceados: {total_linhas}")
    print("-" * 50)
    print("Distribuição das categorias de cores no banco:")
    
    for cor, qtd in sorted(contagem_cores.items()):
        print(f"  • {cor}: {qtd} pontos")
    print("-" * 50)


if __name__ == "__main__":
    gerar_banco_limpo()