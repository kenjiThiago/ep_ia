import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# Função de ativação sigmoide
def sigmoid(entrada):
    # σ(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-entrada))


# Derivada da sigmoide (usada no backpropagation)
def sigmoid_derivada(x):
    # σ'(x) = σ(x) * (1 - σ(x))
    return x * (1 - x)


# Adiciona coluna de bias (1s) ao início da matriz de entradas
def adicionar_bias(matriz_entradas):
    ones = np.ones((matriz_entradas.shape[0], 1))
    return np.hstack((ones, matriz_entradas))


# Inicializa pesos aleatórios com valores pequenos
def inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, num_neuronios_ocultos):
    # pesos da camada escondida: (n_ocultos x n_entradas)
    pesos_camada_escondida = np.random.uniform(-0.01, 0.01, (num_neuronios_ocultos, numero_pesos_escondida))
    # pesos da camada de saída: (n_saídas x (n_ocultos + 1)) → +1 por causa do bias
    pesos_camada_saida = np.random.uniform(-0.01, 0.01, (neuronios_camada_saida, num_neuronios_ocultos + 1))
    return pesos_camada_escondida, pesos_camada_saida


# Propagação direta (forward pass)
def forward_pass(pesos_camada_escondida, pesos_camada_saida, entrada):
    # Ativações da camada oculta (sem bias)
    entradas_camada_escondida = np.dot(pesos_camada_escondida, entrada)
    saida_camada_escondida = sigmoid(entradas_camada_escondida)

    # Adiciona o bias (1) à saída da camada oculta
    saida_camada_escondida = np.insert(saida_camada_escondida, 0, 1)

    # Ativação da camada de saída
    entradas_camada_saida = np.dot(pesos_camada_saida, saida_camada_escondida)
    predicao_final = sigmoid(entradas_camada_saida)

    return saida_camada_escondida, predicao_final


# Backpropagation: atualiza os pesos com base no erro
def backpropagation(
    pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, taxa_aprendizado,
):
    # Cálculo do erro da camada de saída:
    # erro_saida = y - y_predito
    erro_saida = y_i - predicao_final

    # Gradiente da camada de saída:
    # delta_saida = erro_saida * σ'(y_hat)
    delta_saida = erro_saida * sigmoid_derivada(predicao_final)

    # Cálculo da influência dos erros da saída na camada oculta:
    # soma_escondida = W_saida.T * delta_saida  (sem incluir o bias)
    soma_escondida = np.dot(pesos_camada_saida[:, 1:].T, delta_saida)

    # Gradiente da camada oculta:
    # delta_escondida = soma_escondida * σ'(ativação_oculta)
    delta_escondida = soma_escondida * sigmoid_derivada(saida_camada_escondida[1:])

    # Atualização dos pesos com gradiente descendente:
    # W += taxa * delta * entrada_transposta
    pesos_camada_saida += taxa_aprendizado * np.outer(delta_saida, saida_camada_escondida)
    pesos_camada_escondida += taxa_aprendizado * np.outer(delta_escondida, x_i)

    # Erro quadrático total da amostra
    erro = np.sum(erro_saida**2)

    return pesos_camada_escondida, pesos_camada_saida, erro


# Treinamento por várias épocas, com cálculo do erro médio por época
def treinar_epocas(
        x_train, y_train, pesos_camada_escondida, pesos_camada_saida, taxa_aprendizado, epocas, x_valid, y_valid
):
    erros = []
    erros_validacao = []

    menor_erro_validacao = 100
    melhor_epoca = 0
    melhores_pesos_camada_escondida = pesos_camada_escondida
    melhores_pesos_camada_saida = pesos_camada_saida

    for i in range(epocas):
        if (i + 1) % 50 == 0:
            print(f"{i + 1} épocas concluídas")
        erro_total = 0
        for x_i, y_i in zip(x_train, y_train):
            saida_camada_escondida, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)
            pesos_camada_escondida, pesos_camada_saida, erro = backpropagation(
                pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, taxa_aprendizado,
            )

            erro_total += erro

        erro_medio = erro_total / x_train.shape[0]
        erros.append(erro_medio)

        # Validação (opcional)
        if x_valid is not None and y_valid is not None:
            erro_validacao_atual = validacao_rede(x_valid, y_valid, pesos_camada_escondida, pesos_camada_saida)
            erros_validacao.append(erro_validacao_atual)

            if erro_validacao_atual < menor_erro_validacao:
                melhor_epoca = i
                menor_erro_validacao = erro_validacao_atual
                melhores_pesos_camada_escondida = pesos_camada_escondida
                melhores_pesos_camada_saida = pesos_camada_saida
        else:
            melhores_pesos_camada_escondida = pesos_camada_escondida
            melhores_pesos_camada_saida = pesos_camada_saida

    if x_valid is not None and y_valid is not None:
        print(f"\nMelhor época: {melhor_epoca} | Menor erro de validação: {menor_erro_validacao:.6f}")

    return melhores_pesos_camada_escondida, melhores_pesos_camada_saida, erros, erros_validacao


# Gera gráfico do erro durante o treinamento
def plotar_erro(erros, erros_validacao):
    plt.plot(erros, label="Erro")
    if len(erros_validacao) != 0:
        plt.plot(erros_validacao, label="Erro Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotar_confusion_matrix(matriz):
    labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # Letras A-Z

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(matriz, cmap='Blues')
    plt.title("Matriz de Confusão", pad=20)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel("Previsto")
    plt.ylabel("Real")

    # Anota os valores dentro das células
    for i in range(len(labels)):
        for j in range(len(labels)):
            valor = matriz[i, j]
            cor = "white" if valor > matriz.max() / 2 else "black"
            ax.text(j, i, str(valor), ha='center', va='center', color=cor)

    plt.tight_layout()
    plt.show()

# Treinamento padrão (com ou sem validação)
def treinamento(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, x_valid=None, y_valid=None, plotar=True):
    numero_pesos_escondida = x_train.shape[1]
    neuronios_camada_saida = y_train.shape[1]

    pesos_camada_escondida, pesos_camada_saida = inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, num_neuronios_ocultos)

    pesos_camada_escondida, pesos_camada_saida, erros, erros_validacao = treinar_epocas(
        x_train, y_train, pesos_camada_escondida, pesos_camada_saida, taxa_aprendizado, epocas, x_valid, y_valid
    )
    if plotar: plotar_erro(erros, erros_validacao)

    return pesos_camada_escondida, pesos_camada_saida


# Treinamento com validação
def treinamento_validacao(entradas_brutas, saidas_desejadas, taxa_aprendizado, epocas, num_neuronios_ocultos, tamanho_validacao, plotar=True):
    num_amostras = entradas_brutas.shape[0]
    total_saidas = saidas_desejadas.shape[0]

    x_train = entradas_brutas[: (num_amostras - tamanho_validacao)]
    y_train = saidas_desejadas[: (total_saidas - tamanho_validacao)]

    x_valid = entradas_brutas[(num_amostras - tamanho_validacao) :]
    y_valid = saidas_desejadas[(total_saidas - tamanho_validacao) :]

    pesos_camada_escondida, pesos_camada_saida = treinamento(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, x_valid, y_valid, plotar)

    return pesos_camada_escondida, pesos_camada_saida


# Treinamento com K-Fold Cross Validation
def treinamento_folds(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, numero_folds, plotar=True):
    tamanho_treinamento = x_train.shape[0]
    indices = np.arange(tamanho_treinamento)
    np.random.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    folds_x = np.array_split(x_train, numero_folds)
    folds_y = np.array_split(y_train, numero_folds)

    acertos_totais = 0
    total_testes = 0
    acuracias = []

    melhor_acuracia = 0
    melhor_pesos_camada_escondida = None
    melhor_pesos_camada_saida = None
    fold = 0
    for f in range(numero_folds):
        print(f"=======Fold {f+1}/{numero_folds}=======")

        x_fold_valid = folds_x[f]
        y_fold_valid = folds_y[f]

        x_fold_train = np.concatenate([folds_x[j] for j in range(numero_folds) if j != f])
        y_fold_train = np.concatenate([folds_y[j] for j in range(numero_folds) if j != f])

        pesos_camada_escondida, pesos_camada_saida = treinamento(x_fold_train, y_fold_train, taxa_aprendizado, epocas, num_neuronios_ocultos, plotar=plotar)

        acuracia = testar_rede(x_fold_valid, y_fold_valid, pesos_camada_escondida, pesos_camada_saida, False)
        acuracias.append(acuracia)

        acertos_totais += int(acuracia * x_fold_valid.shape[0])
        total_testes += x_fold_valid.shape[0]

        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_pesos_camada_escondida = np.copy(pesos_camada_escondida)
            melhor_pesos_camada_saida = np.copy(pesos_camada_saida)
            fold = f + 1

    media_acuracia = acertos_totais / total_testes
    desvio_padrao = np.std(acuracias, ddof=1)

    print("\n=======Resultados k-fold=======")
    print(f"Fold com melhor acuracia: {fold}")
    print(f"Acurácias por fold:\n{acuracias}")
    print(f"Média da acurácia: {media_acuracia} | Desvio padrão: {desvio_padrao}")

    return melhor_pesos_camada_escondida, melhor_pesos_camada_saida


# Calcula erro quadrático médio em um conjunto de validação
def validacao_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida):
    total = entradas.shape[0]

    erro_total = 0
    for x_i, y_i in zip(entradas, saida_desejada):
        _, predicao_final = forward_pass(
            pesos_camada_escondida, pesos_camada_saida, x_i
        )
        erro_saida = y_i - predicao_final
        erro_total += np.sum(erro_saida**2)

    return erro_total / total


# Testa a rede comparando a classe prevista com a esperada
def testar_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida, print_resultado=True):
    acertos = 0
    total = entradas.shape[0]
    verdadeiras = []
    previstas = []

    for x_i, y_i in zip(entradas, saida_desejada):
        _, predicao_final = forward_pass(
            pesos_camada_escondida, pesos_camada_saida, x_i
        )

        classe_real = np.argmax(y_i)
        classe_prevista = np.argmax(predicao_final)

        verdadeiras.append(classe_real)
        previstas.append(classe_prevista)

        if classe_prevista == classe_real:
            acertos += 1

    acuracia = acertos / total
    if print_resultado:
        print("\n=======Resultados Teste=======")
        print(f"Acurácia final no conjunto de teste: {acuracia} ({acertos}/{total})")

        f1 = f1_score(verdadeiras, previstas, average="macro")
        print(f"F1-score (macro): {f1:.4f}")

        matriz = confusion_matrix(verdadeiras, previstas)
        plotar_confusion_matrix(matriz)

    return acuracia
