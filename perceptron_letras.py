import numpy as np
import perceptron as pc

entradas_brutas = np.load("dados/X.npy")
entradas_brutas = entradas_brutas.reshape(-1, 120)
entradas_brutas = pc.adicionar_bias(entradas_brutas)
num_amostras = entradas_brutas.shape[0]

saidas_desejadas = np.load("dados/Y_classe.npy")
saidas_desejadas = saidas_desejadas.reshape(-1, 26)
total_saidas = saidas_desejadas.shape[0]

taxa_aprendizado = 0.2
epocas = 1000
num_neuronios_ocultos = 80

# pesos_camada_escondida, pesos_camada_saida = pc.treinamento(entradas_brutas, saidas_desejadas, taxa_aprendizado,
#                                                                    epocas, num_neuronios_ocultos)

# pesos_camada_escondida, pesos_camada_saida = pc.treinamento_validacao(entradas_brutas, saidas_desejadas, taxa_aprendizado,
#                                                                                               epocas, num_neuronios_ocultos)

pesos_camada_escondida, pesos_camada_saida = pc.treinamento_folds(entradas_brutas, saidas_desejadas, taxa_aprendizado, epocas, num_neuronios_ocultos, 13)

# x_test = entradas_brutas[(num_amostras - 130):]
# y_test = saidas_desejadas[(total_saidas - 130):]
#
# print(pc.testar_rede(x_test, y_test, pesos_camada_escondida, pesos_camada_saida))
