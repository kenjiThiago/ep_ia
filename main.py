import numpy as np
import perceptron as pc
import argparse
import os


# Faz o parse dos argumentos
def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de rede neural para reconhecimento de letras.")
    parser.add_argument("-x", "--entrada", type=str, required=True, help="Caminho para o arquivo .npy com as entradas (X)")
    parser.add_argument("-y", "--saida", type=str, required=True, help="Caminho para o arquivo .npy com as saídas (Y)")
    parser.add_argument("-m", "--modo", choices=["simples", "validacao", "kfold"], default="simples", help="Modo de treinamento")
    parser.add_argument("-n", "--neuronios", type=int, default=60, help="Número de neurônios na camada oculta")
    parser.add_argument("-t", "--taxa", type=float, default=0.1, help="Taxa de aprendizado")
    parser.add_argument("-e", "--epocas", type=int, default=500, help="Número de épocas de treinamento")
    parser.add_argument("-f", "--folds", type=int, default=10, help="Número de folds (apenas se modo for kfold).")
    parser.add_argument("-p", "--plot", choices=["True", "False"], default="True", help="Mostrar gráfico do erro quadrático médio")
    parser.add_argument("-a", "--ativacao", choices=["sigmoide", "tanh"], default="sigmoide", help="Função de ativação usada na rede neural")
    parser.add_argument("--embaralhar", action="store_true", help="Se definido, embaralha os dados antes do treinamento.")

    args = parser.parse_args()
    return args


# Recebe os argumentos da linha de comando
args = parse_args()

# Verifica se os arquivos existem
if not os.path.exists(args.entrada):
    raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.entrada}")

if not os.path.exists(args.saida):
    raise FileNotFoundError(f"Arquivo de saída não encontrado: {args.saida}")

# Carrega os dados de entrada (features) e os ajusta para ter 120 colunas por amostra
entradas_brutas = np.load(args.entrada)
entradas_brutas = entradas_brutas.reshape(-1, 120)
# Adiciona o termo de bias às entradas (geralmente um 1 no início de cada vetor de entrada)
entradas_brutas = pc.adicionar_bias(entradas_brutas)

# Carrega os rótulos (saídas desejadas com 26 neurônios para A-Z)
saidas_desejadas = np.load(args.saida)

# Define quantas amostras serão usadas para teste (o restante será usado para treinamento)
tamanho_teste = 130
# Obtém o número total de amostras e saídas
num_amostras = entradas_brutas.shape[0]

tamanho_treino = num_amostras - tamanho_teste

# Separa os dados de treinamento (tudo menos as últimas 130 amostras)
x_treino = entradas_brutas[:tamanho_treino]
y_treino = saidas_desejadas[:tamanho_treino]

# Define os parâmetros da rede
hparams = {
    "modo": args.modo,
    "num_neuronios_ocultos": args.neuronios,
    "taxa_aprendizado": args.taxa,
    "epocas": args.epocas,
    "plot": args.plot == "True",
    "folds": args.folds,
    "func_ativacao": pc.FUNCOES_ATIVACAO[args.ativacao][0],
    "func_derivada": pc.FUNCOES_ATIVACAO[args.ativacao][1],
    "embaralhar": args.embaralhar
}

# Usa 18% dos dados de treinamento para a validação
tamanho_validacao = int(0.18 * (tamanho_treino))

modo = hparams["modo"]

# Separa os dados de teste (últimas 130 amostras)
x_teste = entradas_brutas[tamanho_treino:]
y_teste = saidas_desejadas[tamanho_treino:]

if modo == "simples":
    # 1. Treinamento simples com todos os dados de treino (sem validação)
    pesos_camada_escondida, pesos_camada_saida = pc.treinamento(
        x_treino, y_treino, hparams, tipo="simples"
    )

    # Testa a rede com os pesos finais no conjunto de teste e imprime a acurácia
    pc.testar_rede(x_teste, y_teste, pesos_camada_escondida, pesos_camada_saida, hparams["func_ativacao"])
elif modo == "validacao":
    # 2. Treinamento com validação
    # Usa parte dos dados de treino como validação durante as épocas (e.g., últimas N amostras)
    pesos_camada_escondida, pesos_camada_saida = pc.treinamento_validacao(
        x_treino, y_treino, hparams, tamanho_validacao
    )

    # Testa a rede com os pesos finais no conjunto de teste e imprime a acurácia
    pc.testar_rede(x_teste, y_teste, pesos_camada_escondida, pesos_camada_saida, hparams["func_ativacao"])
else:
    # 3. Treinamento com K-Fold Cross Validation
    # Divide o conjunto de treino em K partes, treina com K-1 e valida com 1, repetindo K vezes
    pc.treinamento_folds(
        x_treino, y_treino, hparams
    )
