import numpy as np
import perceptron as pc

entradas_brutas = np.load("dados/X.npy")
entradas_brutas = entradas_brutas.reshape(-1, 120)
entradas_brutas = pc.adicionar_bias(entradas_brutas)
num_amostras = entradas_brutas.shape[0]

saidas_desejadas = np.load("dados/Y_classe.npy")
saidas_desejadas = saidas_desejadas.reshape(-1, 26)
total_saidas = saidas_desejadas.shape[0]

num_amostras = entradas_brutas.shape[0]
total_saidas = saidas_desejadas.shape[0]

tamanho_treinamento = 130

x_train = entradas_brutas[: (num_amostras - tamanho_treinamento)]
y_train = saidas_desejadas[: (total_saidas - tamanho_treinamento)]

taxa_aprendizado = 0.3
epocas = 200
num_neuronios_ocultos = 80

# pesos_camada_escondida, pesos_camada_saida = pc.treinamento(
#     x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos
# )

# pesos_camada_escondida, pesos_camada_saida = pc.treinamento_validacao(
#     x_train, y_train, taxa_aprendizado, epocas, num_neuronios_ocultos, 130
# )
#
pesos_camada_escondida, pesos_camada_saida = pc.treinamento_folds(
    x_train,
    y_train,
    taxa_aprendizado,
    epocas,
    num_neuronios_ocultos,
    13,
)

x_test = entradas_brutas[(num_amostras - tamanho_treinamento) :]
y_test = saidas_desejadas[(total_saidas - tamanho_treinamento) :]

print(pc.testar_rede(x_test, y_test, pesos_camada_escondida, pesos_camada_saida))
