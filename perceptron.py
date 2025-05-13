import numpy as np
import matplotlib.pyplot as plt


def sigmoid(entrada):
    return 1 / (1 + np.exp(-entrada))


def sigmoid_derivada(x):
    return x * (1 - x)


def adicionar_bias(matriz_entradas):
    ones = np.ones((matriz_entradas.shape[0], 1))
    return np.hstack((ones, matriz_entradas))


def inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, num_neuronios_ocultos):
    pesos_camada_escondida = np.random.uniform(-0.01, 0.01, (num_neuronios_ocultos, numero_pesos_escondida))
    pesos_camada_saida = np.random.uniform(-0.01, 0.01, (neuronios_camada_saida, num_neuronios_ocultos + 1))
    return pesos_camada_escondida, pesos_camada_saida


def forward_pass(pesos_camada_escondida, pesos_camada_saida, entrada):
    entradas_camada_escondida = np.dot(pesos_camada_escondida, entrada)
    saida_camada_escondida = sigmoid(entradas_camada_escondida)

    # Adiciona o bias na posição 0 da lista
    saida_camada_escondida = np.insert(saida_camada_escondida, 0, 1)

    entradas_camada_saida = np.dot(pesos_camada_saida, saida_camada_escondida)
    predicao_final = sigmoid(entradas_camada_saida)

    return saida_camada_escondida, predicao_final


def backpropagation(
    pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, taxa_aprendizado,
):
    erro_saida = y_i - predicao_final
    delta_saida = erro_saida * sigmoid_derivada(predicao_final)

    soma_escondida = np.dot(pesos_camada_saida[:, 1:].T, delta_saida)
    delta_escondida = soma_escondida * sigmoid_derivada(saida_camada_escondida[1:])

    pesos_camada_saida += taxa_aprendizado * np.outer(delta_saida, saida_camada_escondida)

    pesos_camada_escondida += taxa_aprendizado * np.outer(delta_escondida, x_i)

    erro = np.sum(erro_saida**2)

    return pesos_camada_escondida, pesos_camada_saida, erro


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
        if i % 100 == 0:
            print(f"Época {i} concluída")
        erro_total = 0
        for x_i, y_i in zip(x_train, y_train):
            saida_camada_escondida, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)
            pesos_camada_escondida, pesos_camada_saida, erro = backpropagation(
                pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, taxa_aprendizado,
            )

            erro_total += erro

        erro_medio = erro_total / x_train.shape[0]
        erros.append(erro_medio)

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
        print(melhor_epoca, menor_erro_validacao)

    return melhores_pesos_camada_escondida, melhores_pesos_camada_saida, erros, erros_validacao


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


def treinamento(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, x_valid=None, y_valid=None, plotar=True):
    numero_pesos_escondida = x_train.shape[1]
    neuronios_camada_saida = y_train.shape[1]

    pesos_camada_escondida, pesos_camada_saida = inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, num_neuronios_ocultos)

    pesos_camada_escondida, pesos_camada_saida, erros, erros_validacao = treinar_epocas(
        x_train, y_train, pesos_camada_escondida, pesos_camada_saida, taxa_aprendizado, epocas, x_valid, y_valid
    )
    if plotar: plotar_erro(erros, erros_validacao)

    return pesos_camada_escondida, pesos_camada_saida


def treinamento_validacao(entradas_brutas, saidas_desejadas, taxa_aprendizado, epocas, num_neuronios_ocultos, tamanho_validacao, plotar=True):
    num_amostras = entradas_brutas.shape[0]
    total_saidas = saidas_desejadas.shape[0]

    x_train = entradas_brutas[: (num_amostras - tamanho_validacao)]
    y_train = saidas_desejadas[: (total_saidas - tamanho_validacao)]

    x_valid = entradas_brutas[(num_amostras - tamanho_validacao) :]
    y_valid = saidas_desejadas[(total_saidas - tamanho_validacao) :]

    pesos_camada_escondida, pesos_camada_saida = treinamento(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, x_valid, y_valid, plotar)

    return pesos_camada_escondida, pesos_camada_saida


def treinamento_folds(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, numero_folds, plotar=True):
    tamanho_treinamento = x_train.shape[0]
    fold_size = tamanho_treinamento // numero_folds

    melhor_acuracia = 0
    acuracias = []

    melhor_pesos_camada_escondida = None
    melhor_pesos_camada_saida = None
    fold = 0
    for f in range(numero_folds):
        print(f"Fold {f}")

        inicio = f * fold_size
        fim = inicio + fold_size if f < numero_folds - 1 else tamanho_treinamento

        x_fold_valid = x_train[inicio:fim]
        y_fold_valid = y_train[inicio:fim]

        x_fold_train = np.concatenate((x_train[:inicio], x_train[fim:]), axis=0)
        y_fold_train = np.concatenate((y_train[:inicio], y_train[fim:]), axis=0)

        pesos_camada_escondida, pesos_camada_saida = treinamento(x_fold_train, y_fold_train, taxa_aprendizado, epocas, num_neuronios_ocultos, plotar=plotar)
        acuracia = testar_rede(x_fold_valid, y_fold_valid, pesos_camada_escondida, pesos_camada_saida)

        acuracias.append(acuracia)
        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_pesos_camada_escondida = np.copy(pesos_camada_escondida)
            melhor_pesos_camada_saida = np.copy(pesos_camada_saida)
            fold = f

    print(fold)
    print(acuracias)
    print(np.mean(acuracias), np.std(acuracias, ddof=1))

    return melhor_pesos_camada_escondida, melhor_pesos_camada_saida


def validacao_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida):
    total = entradas.shape[0]

    erro_total = 0
    for i in range(total):
        x_i = entradas[i]
        y_i = saida_desejada[i]

        _, predicao_final = forward_pass(
            pesos_camada_escondida, pesos_camada_saida, x_i
        )
        erro_saida = y_i - predicao_final
        erro_total += np.sum(erro_saida**2)

    return erro_total / total


def testar_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida):
    acertos = 0
    total = entradas.shape[0]

    for i in range(total):
        x_i = entradas[i]
        y_i = saida_desejada[i]

        _, predicao_final = forward_pass(
            pesos_camada_escondida, pesos_camada_saida, x_i
        )

        pred_binaria = np.zeros_like(predicao_final)
        pred_binaria[np.argmax(predicao_final)] = 1

        if np.argmax(pred_binaria) == np.argmax(y_i):
            acertos += 1

    acuracia = acertos / total
    return acuracia
