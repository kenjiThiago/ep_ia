import numpy as np

def tanh(entrada):
    return np.tanh(entrada)

def tanh_derivada(x):
    return 1 - x**2

def sigmoid(entrada):
    return 1 / (1 + np.exp(-entrada))

def sigmoid_derivada(x):
    return x * (1 - x)

def inicia_pesos(numero_pesos_escondida, neuronios_camada_escondida, neuronios_camada_saida):
    pesos_camada_escondida = np.random.uniform(-0.01, 0.01, (neuronios_camada_escondida, numero_pesos_escondida))
    pesos_camada_saida = np.random.uniform(-0.01, 0.01, (neuronios_camada_saida, neuronios_camada_escondida + 1))
    return pesos_camada_escondida, pesos_camada_saida

def forward_pass(pesos_camada_escondida, pesos_camada_saida, entrada):
    entradas_camada_escondida = np.dot(pesos_camada_escondida, entrada)
    saida_camada_escondida = sigmoid(entradas_camada_escondida)

    # Adiciona o bias na posição 0 da lista
    saida_camada_escondida = np.insert(saida_camada_escondida, 0, 1)

    entradas_camada_saida = np.dot(pesos_camada_saida, saida_camada_escondida)
    predicao_final = sigmoid(entradas_camada_saida)

    return saida_camada_escondida, predicao_final

def treinamento_validacao(entradas, saidas_desejadas, taxa_aprendizado, epocas, numero_neuronios, entradas_validacao, saidas_validacao):
    numero_pesos_escondida = len(entradas[0])
    neuronios_camada_saida = len(saidas_desejadas[0])
    numero_entradas = len(entradas)

    pesos_camada_escondida, pesos_camada_saida = inicia_pesos(numero_pesos_escondida, numero_neuronios, neuronios_camada_saida)

    erros = []
    erros_validacao = []
    for i in range(epocas):
        if i % 100 == 0: print(f"Época {i} concluída")
        erro_total = 0
        for j in range(numero_entradas):
            x_i = entradas[j]
            y_i = saidas_desejadas[j]

            saida_camada_escondida, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)

            erro_saida = y_i - predicao_final
            delta_saida = erro_saida * sigmoid_derivada(predicao_final)

            soma_escondida = np.dot(pesos_camada_saida[:, 1:].T, delta_saida)
            delta_escondida = soma_escondida * sigmoid_derivada(saida_camada_escondida[1:])

            pesos_camada_saida += taxa_aprendizado * np.outer(delta_saida, saida_camada_escondida)

            pesos_camada_escondida += taxa_aprendizado * np.outer(delta_escondida, x_i)

            erro_total += np.sum(erro_saida ** 2)
        erros.append(erro_total / (numero_entradas))
        erros_validacao.append(validacao_rede(entradas_validacao, saidas_validacao, pesos_camada_escondida, pesos_camada_saida))

    return pesos_camada_escondida, pesos_camada_saida, erros, erros_validacao

def treinamento(entradas, saidas_desejadas, taxa_aprendizado, epocas, numero_neuronios):
    numero_pesos_escondida = len(entradas[0])
    neuronios_camada_saida = len(saidas_desejadas[0])
    numero_entradas = len(entradas)

    pesos_camada_escondida, pesos_camada_saida = inicia_pesos(numero_pesos_escondida, numero_neuronios, neuronios_camada_saida)

    erros = []
    for i in range(epocas):
        if i % 100 == 0: print(f"Época {i} concluída")
        erro_total = 0
        for j in range(numero_entradas):
            x_i = entradas[j]
            y_i = saidas_desejadas[j]

            saida_camada_escondida, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)

            erro_saida = y_i - predicao_final
            delta_saida = erro_saida * sigmoid_derivada(predicao_final)

            soma_escondida = np.dot(pesos_camada_saida[:, 1:].T, delta_saida)
            delta_escondida = soma_escondida * sigmoid_derivada(saida_camada_escondida[1:])

            pesos_camada_saida += taxa_aprendizado * np.outer(delta_saida, saida_camada_escondida)

            pesos_camada_escondida += taxa_aprendizado * np.outer(delta_escondida, x_i)

            erro_total += np.sum(erro_saida ** 2)
        erros.append(erro_total / (numero_entradas))

    return pesos_camada_escondida, pesos_camada_saida, erros

def validacao_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida):
    total = len(entradas)

    erro_total = 0
    numero_entradas = len(entradas)
    for i in range(total):
        x_i = entradas[i]
        y_i = saida_desejada[i]

        _, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)
        erro_saida = y_i - predicao_final
        erro_total += np.sum(erro_saida ** 2)

    return erro_total / numero_entradas


def testar_rede(entradas, saida_desejada, pesos_camada_escondida, pesos_camada_saida):
    acertos = 0
    total = len(entradas)

    for i in range(total):
        x_i = entradas[i]
        y_i = saida_desejada[i]

        _, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)

        pred_binaria = np.zeros_like(predicao_final)
        pred_binaria[np.argmax(predicao_final)] = 1

        if np.argmax(pred_binaria) == np.argmax(y_i):
            acertos += 1

    acuracia = acertos / total
    return acuracia

def adicionar_bias(matriz_entradas):
    ones = np.ones((matriz_entradas.shape[0], 1))
    return np.hstack((ones, matriz_entradas))
