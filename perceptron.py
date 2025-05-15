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
    # sigma_saida = erro_saida * σ'(y_hat)
    sigma_saida = erro_saida * sigmoid_derivada(predicao_final)

    # Cálculo da influência dos erros da saída na camada oculta:
    # soma_escondida = W_saida.T * sigma_saida  (sem incluir o bias)
    soma_escondida = np.dot(pesos_camada_saida[:, 1:].T, sigma_saida)

    # Gradiente da camada oculta:
    # sigma_escondida = soma_escondida * σ'(ativação_oculta)
    sigma_escondida = soma_escondida * sigmoid_derivada(saida_camada_escondida[1:])

    # Atualização dos pesos com gradiente descendente:
    # W += taxa * sigma * entrada_transposta
    pesos_camada_saida += taxa_aprendizado * np.outer(sigma_saida, saida_camada_escondida)
    pesos_camada_escondida += taxa_aprendizado * np.outer(sigma_escondida, x_i)

    # Erro quadrático total da amostra
    erro = np.sum(erro_saida**2)

    return pesos_camada_escondida, pesos_camada_saida, erro


# Treinamento por várias épocas, com cálculo do erro médio por época
def treinar_epocas(
        x_train, y_train, pesos_camada_escondida, pesos_camada_saida, taxa_aprendizado, epocas, x_valid, y_valid
):
    # Lista para armazenar o erro quadrático médio (EQM) de cada época
    erros = []
    # Lista para armazenar o erro de validação, caso fornecido
    erros_validacao = []

    # Inicializa o menor erro de validação com um valor alto
    menor_erro_validacao = 100

    # Inicializa os melhores pesos e a melhor época
    melhor_epoca = 0
    melhores_pesos_camada_escondida = pesos_camada_escondida
    melhores_pesos_camada_saida = pesos_camada_saida

    # Loop principal de treinamento
    for i in range(epocas):
        if (i + 1) % 50 == 0:
            print(f"{i + 1} épocas concluídas")
        erro_total = 0

        # Itera por todas as amostras de treino
        for x_i, y_i in zip(x_train, y_train):
            saida_camada_escondida, predicao_final = forward_pass(pesos_camada_escondida, pesos_camada_saida, x_i)
            pesos_camada_escondida, pesos_camada_saida, erro = backpropagation(
                pesos_camada_escondida, pesos_camada_saida, x_i, y_i, saida_camada_escondida, predicao_final, taxa_aprendizado,
            )

            # Acumula o erro da amostra
            erro_total += erro

        # Calcula o erro médio da época atual e armazena
        erro_medio = erro_total / x_train.shape[0]
        erros.append(erro_medio)

        # Validação, se fornecida
        if x_valid is not None and y_valid is not None:
            # Calcula o erro quadrático médio no conjunto de validação
            erro_validacao_atual = validacao_rede(x_valid, y_valid, pesos_camada_escondida, pesos_camada_saida)
            erros_validacao.append(erro_validacao_atual)

            # Atualiza os melhores pesos se o erro de validação foi o menor até agora
            if erro_validacao_atual < menor_erro_validacao:
                melhor_epoca = i
                menor_erro_validacao = erro_validacao_atual
                melhores_pesos_camada_escondida = pesos_camada_escondida
                melhores_pesos_camada_saida = pesos_camada_saida
        else:
            # Se não há validação, os melhores pesos são os atuais
            melhores_pesos_camada_escondida = pesos_camada_escondida
            melhores_pesos_camada_saida = pesos_camada_saida
        # Critério de parada antecipada: se o erro for muito pequeno
        if erro_medio < 5e-3:
            break

    # Imprime o erro final de treino
    print(f"\nErro Quadrático Médio Final: {erros[-1]}")

    # Se houver validação, exibe a melhor época e o menor erro de validação
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

    # Inicializa os pesos das camadas
    pesos_camada_escondida, pesos_camada_saida = inicia_pesos(numero_pesos_escondida, neuronios_camada_saida, num_neuronios_ocultos)

    # Executa o treinamento por múltiplas épocas
    # A função 'treinar_epocas' cuida do forward, backpropagation e validação (se houver)
    pesos_camada_escondida, pesos_camada_saida, erros, erros_validacao = treinar_epocas(
        x_train, y_train, pesos_camada_escondida, pesos_camada_saida, taxa_aprendizado, epocas, x_valid, y_valid
    )

    # Se habilitado, plota o gráfico do erro por época
    if plotar: plotar_erro(erros, erros_validacao)

    return pesos_camada_escondida, pesos_camada_saida


# Treinamento com validação
def treinamento_validacao(entradas_brutas, saidas_desejadas, taxa_aprendizado, epocas, num_neuronios_ocultos, tamanho_validacao, plotar=True):
    num_amostras = entradas_brutas.shape[0]
    total_saidas = saidas_desejadas.shape[0]

    # Separa os dados em treino e validação
    # Os últimos 'tamanho_validacao' exemplos são usados como validação
    x_train = entradas_brutas[: (num_amostras - tamanho_validacao)]
    y_train = saidas_desejadas[: (total_saidas - tamanho_validacao)]

    x_valid = entradas_brutas[(num_amostras - tamanho_validacao) :]
    y_valid = saidas_desejadas[(total_saidas - tamanho_validacao) :]

    # Treina a rede com os dados de treino e valida com os dados separados
    pesos_camada_escondida, pesos_camada_saida = treinamento(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, x_valid, y_valid, plotar)

    return pesos_camada_escondida, pesos_camada_saida


# Treinamento com K-Fold Cross Validation
def treinamento_folds(x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, numero_folds, plotar=True):
    tamanho_treinamento = x_train.shape[0]

    # Gera uma permutação aleatória dos índices (embaralhamento dos dados)
    indices = np.arange(tamanho_treinamento)
    np.random.shuffle(indices)

    # Aplica a permutação às entradas e saídas
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Divide os dados embaralhados em 'numero_folds' partes iguais (ou quase iguais)
    folds_x = np.array_split(x_train, numero_folds)
    folds_y = np.array_split(y_train, numero_folds)

    # Variáveis para computar estatísticas globais
    acertos_totais = 0          # Soma de acertos de todos os folds
    total_testes = 0            # Total de amostras testadas
    acuracias = []              # Lista com acurácia de cada fold

    todos_reais = []            # Lista com todas as classes reais de todos os folds
    todos_previstos = []        # Lista com todas as classes previstas pela rede

    # Loop principal dos folds
    for f in range(numero_folds):
        print(f"=======Fold {f+1}/{numero_folds}=======")

        # Separa os dados de validação (o fold atual)
        x_fold_valid = folds_x[f]
        y_fold_valid = folds_y[f]

        # Concatena os demais folds para formar o conjunto de treino
        x_fold_train = np.concatenate([folds_x[j] for j in range(numero_folds) if j != f])
        y_fold_train = np.concatenate([folds_y[j] for j in range(numero_folds) if j != f])

        # Treina a rede com os dados do fold atual
        pesos_camada_escondida, pesos_camada_saida = treinamento(x_fold_train, y_fold_train, taxa_aprendizado, epocas, num_neuronios_ocultos, plotar=plotar)

        # Avalia a rede no fold de validação
        acuracia, reais, previstos = testar_rede(x_fold_valid, y_fold_valid, pesos_camada_escondida, pesos_camada_saida, False)
        acuracias.append(acuracia)

        # Acumula os rótulos reais e as predições para métricas globais
        todos_reais += reais
        todos_previstos += previstos

        # Acumula estatísticas de acerto
        acertos_totais += int(acuracia * x_fold_valid.shape[0])
        total_testes += x_fold_valid.shape[0]

    # Calcula estatísticas finais após todos os folds
    media_acuracia = acertos_totais / total_testes
    desvio_padrao = np.std(acuracias, ddof=1)

    print("\n=======Resultados k-fold=======")

    print(f"Acurácias por fold:\n{acuracias}")
    print(f"Média da acurácia: {media_acuracia} ({acertos_totais}/{total_testes}) | Desvio padrão: {desvio_padrao}")

    # F1-score global macro (avalia equilíbrio entre precisão e revocação)
    f1 = f1_score(todos_reais, todos_previstos, average="macro")
    print(f"F1-score (macro): {f1:.4f}")

    # Matriz de confusão global
    matriz = confusion_matrix(todos_reais, todos_previstos)
    plotar_confusion_matrix(matriz)


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

    return acuracia, verdadeiras, previstas
