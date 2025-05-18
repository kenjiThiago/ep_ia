import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def tanh(entrada):

    return np.tanh(entrada)


def tanh_derivada(x):

    return 1 - x**2


# Função de ativação sigmoide
def sigmoide(entrada):
    # σ(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-entrada))


# Derivada da sigmoide (usada no backpropagation)
def sigmoide_derivada(x):
    # σ'(x) = σ(x) * (1 - σ(x))
    return x * (1 - x)

FUNCOES_ATIVACAO = {
    "sigmoide": (sigmoide, sigmoide_derivada),
    "tanh": (tanh, tanh_derivada),
}

# Adiciona coluna de bias (1s) ao início da matriz de entradas
def adicionar_bias(matriz_entradas):
    ones = np.ones((matriz_entradas.shape[0], 1))
    return np.hstack((ones, matriz_entradas))


# Inicializa pesos aleatórios com valores pequenos
def inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, num_neuronios_ocultos):
    # pesos da camada escondida: (n_ocultos x n_entradas)
    pesos_camada_escondida = np.random.uniform(-0.1, 0.1, (num_neuronios_ocultos, numero_pesos_escondida))
    # pesos da camada de saída: (n_saídas x (n_ocultos + 1)) → +1 por causa do bias
    pesos_camada_saida = np.random.uniform(-0.1, 0.1, (neuronios_camada_saida, num_neuronios_ocultos + 1))
    return pesos_camada_escondida, pesos_camada_saida


# Propagação direta (forward pass)
def forward_pass(pesos_camada_escondida, pesos_camada_saida, entrada, func_ativacao):
    # Ativações da camada oculta (sem bias)
    entradas_camada_escondida = np.dot(pesos_camada_escondida, entrada)
    saida_camada_escondida = func_ativacao(entradas_camada_escondida)

    # Adiciona o bias (1) à saída da camada oculta
    saida_camada_escondida = np.insert(saida_camada_escondida, 0, 1)

    # Ativação da camada de saída
    entradas_camada_saida = np.dot(pesos_camada_saida, saida_camada_escondida)
    predicao_final = func_ativacao(entradas_camada_saida)

    return saida_camada_escondida, predicao_final


# Backpropagation: atualiza os pesos com base no erro
def backpropagation(
    pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, hparams,
):
    func_derivada = hparams["func_derivada"]
    taxa_aprendizado = hparams["taxa_aprendizado"]

    # Cálculo do erro da camada de saída:
    # erro_saida = y - y_predito
    erro_saida = y_i - predicao_final

    # Gradiente da camada de saída:
    # sigma_saida = erro_saida * σ'(y_hat)
    sigma_saida = erro_saida * func_derivada(predicao_final)

    # Cálculo da influência dos erros da saída na camada oculta:
    # soma_escondida = W_saida.T * sigma_saida  (sem incluir o bias)
    soma_escondida = np.dot(pesos_camada_saida[:, 1:].T, sigma_saida)

    # Gradiente da camada oculta:
    # sigma_escondida = soma_escondida * σ'(ativação_oculta)
    sigma_escondida = soma_escondida * func_derivada(saida_camada_escondida[1:])

    # Atualização dos pesos com gradiente descendente:
    # W += taxa * sigma * entrada_transposta
    pesos_camada_saida += taxa_aprendizado * np.outer(sigma_saida, saida_camada_escondida)
    pesos_camada_escondida += taxa_aprendizado * np.outer(sigma_escondida, x_i)

    # Erro quadrático total da amostra
    erro_quadratico = np.sum(erro_saida**2)

    return pesos_camada_escondida, pesos_camada_saida, erro_quadratico


# Treinamento por várias épocas, com cálculo do erro médio por época
def treinar_epocas(
        x_treino, y_treino, pesos_camada_escondida, pesos_camada_saida, hparams, x_validacao, y_validacao
):
    epocas = hparams["epocas"]

    # Em caso de validacao
    # Lista para armazenar o eqms de validação, caso fornecido
    eqms_validacao = []
    # Inicializa o menor eqm de validação com um valor alto
    menor_eqm = float("inf")
    # Inicializa os melhores pesos e a melhor época
    epoca_parada_antecipada = 0

    # Lista para armazenar o erro quadrático médio (EQM) de cada época
    eqms_treino = []

    pesos_camada_escondida_final = pesos_camada_escondida
    pesos_camada_saida_final = pesos_camada_saida

    # Loop principal de treinamento
    for i in range(epocas):
        if (i + 1) % 50 == 0:
            print(f"{i + 1} épocas concluídas")
        erro_quadratico_total = 0

        # Itera por todas as amostras de treino
        for x_i, y_i in zip(x_treino, y_treino):
            saida_camada_escondida, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i, hparams["func_ativacao"])
            pesos_camada_escondida, pesos_camada_saida, erro_quadratico = backpropagation(
                pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, hparams
            )

            # Acumula o erro quadrático da época
            erro_quadratico_total += erro_quadratico

        # Calcula o eqm da época atual e armazena
        eqm = erro_quadratico_total / x_treino.shape[0]
        eqms_treino.append(eqm)

        # Validação, se fornecida
        if x_validacao is not None and y_validacao is not None:
            # Calcula o erro quadrático médio no conjunto de validação
            eqm_validacao_atual = validacao_rede(x_validacao, y_validacao, pesos_camada_escondida, pesos_camada_saida, hparams["func_ativacao"])
            eqms_validacao.append(eqm_validacao_atual)

            # Atualiza os melhores pesos se o eqm de validação foi o menor até agora
            if eqm_validacao_atual < menor_eqm:
                epoca_parada_antecipada = i + 1
                menor_eqm = eqm_validacao_atual
                pesos_camada_escondida_final = np.copy(pesos_camada_escondida)
                pesos_camada_saida_final = np.copy(pesos_camada_saida)


        # Critério de parada: se o erro for muito pequeno (caso não tenha conjunto de validação)
        elif eqm < 5e-3:
            break

    # Imprime o eqm final de treino
    print(f"\nErro Quadrático Médio Final: {eqms_treino[-1]}")

    # Se houver validação, exibe a melhor época e o menor eqm de validação
    if x_validacao is not None and y_validacao is not None:
        print(f"\nÉpoca da parada antecipada: {epoca_parada_antecipada} | Erro de validação: {menor_eqm:.6f}")

    return pesos_camada_escondida_final, pesos_camada_saida_final, eqms_treino, eqms_validacao, epoca_parada_antecipada


def f1_macro(matriz):
    num_classes = matriz.shape[0]
    f_score = []

    for i in range(num_classes):
        VP = matriz[i, i]               # Verdadeiro Positivo
        FP = np.sum(matriz[:, i]) - VP  # Falso Positivo
        FN = np.sum(matriz[i, :]) - VP  # Falso Negativo

        if (VP + FP) == 0 or (VP + FN) == 0:
            f_score.append(0.0)  # evita divisão por zero
            continue

        precisao = VP / (VP + FP)
        revocacao = VP / (VP + FN)

        if (precisao + revocacao) == 0:
            f_score.append(0.0)
            continue

        f1 = 2 * precisao * revocacao / (precisao + revocacao)
        f_score.append(f1)

    return np.mean(f_score)

# Gera gráfico do erro durante o treinamento
def plotar_erro(eqm_treino, eqm_validacao, epoca_parada_antecipada, hparams):
    plt.plot(eqm_treino, label="Erro")
    if len(eqm_validacao) != 0:
        plt.plot(eqm_validacao, label="Erro Validação")
        plt.axvline(epoca_parada_antecipada, color='black', linestyle='--', label="Época parada antecipada")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio (MSE)")
    plt.legend()
    plt.title(f"Treinamento — TA: {hparams['taxa_aprendizado']} | Épocas: {hparams['epocas']} | Ocultos: {hparams['num_neuronios_ocultos']} | Ativação: {hparams['func_ativacao'].__name__}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotar_matriz_confusao(matriz):
    legendas = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # Letras A-Z

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(matriz, cmap='Blues')
    plt.title("Matriz de Confusão", pad=20)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(legendas)))
    ax.set_yticks(np.arange(len(legendas)))
    ax.set_xticklabels(legendas)
    ax.set_yticklabels(legendas)

    plt.xlabel("Previsto")
    plt.ylabel("Real")

    # Anota os valores dentro das células
    for i in range(len(legendas)):
        for j in range(len(legendas)):
            valor = matriz[i, j]
            cor = "white" if valor > matriz.max() / 2 else "black"
            ax.text(j, i, str(valor), ha='center', va='center', color=cor)

    plt.tight_layout()
    plt.show()

def exibir_parametros(hparams, tipo="simples"):
    print(f"=== Iniciando treinamento ({tipo}) ===")
    print(f"  • Taxa de aprendizado: {hparams['taxa_aprendizado']}")
    print(f"  • Épocas: {hparams['epocas']}")
    print(f"  • Neurônios ocultos: {hparams['num_neuronios_ocultos']}")
    print(f"  • Função de ativação: {hparams['func_ativacao'].__name__}")
    if tipo == "k-fold": print(f"  • Folds: {hparams['folds']}")
    print("=====================================\n")

# Treinamento padrão (com ou sem validação)
def treinamento(x_treino, y_treino, hparams, x_validacao=None, y_validacao=None, tipo=None):
    if tipo:
        exibir_parametros(hparams, tipo)

    plot = hparams["plot"]

    numero_pesos_escondida = x_treino.shape[1]
    neuronios_camada_saida = y_treino.shape[1]

    # Inicializa os pesos das camadas
    pesos_camada_escondida, pesos_camada_saida = inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, hparams["num_neuronios_ocultos"])

    # Executa o treinamento por múltiplas épocas
    # A função 'treinar_epocas' cuida do forward, backpropagation e validação (se houver)
    pesos_camada_escondida, pesos_camada_saida, eqm_treino, eqm_validacao, epoca_parada_antecipada = treinar_epocas(
        x_treino, y_treino, pesos_camada_escondida, pesos_camada_saida, hparams, x_validacao, y_validacao
    )

    # Se habilitado, plota o gráfico do erro por época
    if plot: plotar_erro(eqm_treino, eqm_validacao, epoca_parada_antecipada, hparams)

    return pesos_camada_escondida, pesos_camada_saida


# Treinamento com validação
def treinamento_validacao(entradas_brutas, saidas_desejadas, hparams, tamanho_validacao):
    num_amostras = entradas_brutas.shape[0]
    total_saidas = saidas_desejadas.shape[0]

    # Separa os dados em treino e validação
    # Os últimos 'tamanho_validacao' exemplos são usados como validação
    x_treino = entradas_brutas[: (num_amostras - tamanho_validacao)]
    y_treino = saidas_desejadas[: (total_saidas - tamanho_validacao)]

    x_validacao = entradas_brutas[(num_amostras - tamanho_validacao) :]
    y_validacao = saidas_desejadas[(total_saidas - tamanho_validacao) :]

    # Treina a rede com os dados de treino e valida com os dados separados
    pesos_camada_escondida, pesos_camada_saida = treinamento(x_treino, y_treino, hparams, x_validacao, y_validacao, tipo="validação")

    return pesos_camada_escondida, pesos_camada_saida


# Treinamento com K-Fold Cross Validation
def treinamento_folds(x_treino, y_treino, hparams):
    exibir_parametros(hparams, "k-fold")
    numero_folds = hparams["folds"]

    tamanho_treinamento = x_treino.shape[0]

    np.random.seed(42)
    # Gera uma permutação aleatória dos índices (embaralhamento dos dados)
    indices = np.arange(tamanho_treinamento)
    np.random.shuffle(indices)

    # Aplica a permutação às entradas e saídas
    x_treino = x_treino[indices]
    y_treino = y_treino[indices]

    # Divide os dados embaralhados em 'numero_folds' partes iguais (ou quase iguais)
    folds_x = np.array_split(x_treino, numero_folds)
    folds_y = np.array_split(y_treino, numero_folds)

    acuracias = []              # Lista com acurácia de cada fold

    todos_reais = []            # Lista com todas as classes reais de todos os folds
    todos_previstos = []        # Lista com todas as classes previstas pela rede

    # Loop principal dos folds
    for f in range(numero_folds):
        print(f"=======Fold {f+1}/{numero_folds}=======")

        # Separa os dados de teste (o fold atual)
        x_fold_teste = folds_x[f]
        y_fold_teste = folds_y[f]

        # Concatena os demais folds para formar o conjunto de treino
        x_fold_treino = np.concatenate([folds_x[j] for j in range(numero_folds) if j != f])
        y_fold_treino = np.concatenate([folds_y[j] for j in range(numero_folds) if j != f])

        # Treina a rede com os dados do fold atual
        pesos_camada_escondida, pesos_camada_saida = treinamento(x_fold_treino, y_fold_treino, hparams)

        # Avalia a rede no fold de validação
        acuracia, reais, previstos = testar_rede(x_fold_teste, y_fold_teste, pesos_camada_escondida, pesos_camada_saida, hparams["func_ativacao"], False)
        acuracias.append(acuracia)

        # Acumula os rótulos reais e as predições para métricas globais
        todos_reais += reais
        todos_previstos += previstos

    # Calcula estatísticas finais após todos os folds
    matriz = confusion_matrix(todos_reais, todos_previstos)
    acertos_totais = np.trace(matriz)           # Soma de acertos de todos os folds
    total_testes = np.sum(matriz)               # Total de amostras testadas

    media_acuracia = acertos_totais / total_testes
    desvio_padrao = np.std(acuracias, ddof=1)

    print("\n=======Resultados k-fold=======")

    print(f"Acurácias por fold:\n{acuracias}")
    print(f"Média da acurácia: {media_acuracia} ({acertos_totais}/{total_testes}) | Desvio padrão: {desvio_padrao}")

    # F1-score global macro (avalia equilíbrio entre precisão e revocação)
    f1 = f1_macro(matriz)
    print(f"F1-score (macro): {f1:.4f}")

    # Matriz de confusão global
    plotar_matriz_confusao(matriz)


# Calcula erro quadrático médio em um conjunto de validação
def validacao_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida, func_ativacao):
    total = entradas.shape[0]

    erro_quadratico_total = 0
    for x_i, y_i in zip(entradas, saida_desejada):
        _, predicao_final = forward_pass(
            pesos_camada_escondida, pesos_camada_saida, x_i, func_ativacao
        )
        erro_saida = y_i - predicao_final
        erro_quadratico_total += np.sum(erro_saida**2)

    eqm = erro_quadratico_total / total
    return eqm


# Testa a rede comparando a classe prevista com a esperada
def testar_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida, func_ativacao, print_resultado=True):
    reais = []
    previstos = []

    for x_i, y_i in zip(entradas, saida_desejada):
        _, predicao_final = forward_pass(
            pesos_camada_escondida, pesos_camada_saida, x_i, func_ativacao
        )

        classe_real = np.argmax(y_i)
        classe_prevista = np.argmax(predicao_final)

        reais.append(classe_real)
        previstos.append(classe_prevista)

    # Calcula a matriz de confusão
    matriz = confusion_matrix(reais, previstos)
    acertos = np.trace(matriz)      # Calcula os acertos somando os valores da diagonal principal
    total = np.sum(matriz)
    acuracia = float(acertos / total)

    if print_resultado:
        print("\n=======Resultados Teste=======")
        print(f"Acurácia final no conjunto de teste: {acuracia} ({acertos}/{total})")

        f1 = f1_macro(matriz)
        print(f"F1-score (macro): {f1:.4f}")

        plotar_matriz_confusao(matriz)

    return acuracia, reais, previstos
