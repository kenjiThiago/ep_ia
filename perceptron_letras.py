import numpy as np
import matplotlib.pyplot as plt
import perceptron as pc

entradas_brutas = np.load("dados/X.npy")
entradas_brutas = entradas_brutas.reshape(-1, 120)
entradas_brutas = pc.adicionar_bias(entradas_brutas)
num_amostras = len(entradas_brutas)

saidas_desejadas = np.load("dados/Y_classe.npy")
saidas_desejadas = saidas_desejadas.reshape(-1, 26)
total_saidas = len(saidas_desejadas)

x_train = entradas_brutas[:(num_amostras - 260)]
y_train = saidas_desejadas[:(total_saidas - 260)]

# letras_treinamento = letras[:(total_entradas - 130)]
# saidas_treinamento = saidas[:(total_saidas - 130)]

taxa_aprendizado = 0.2
epocas = 1000
num_neuronios_ocultos = 60

x_valid = entradas_brutas[(num_amostras - 260):(num_amostras - 130)]
y_valid = saidas_desejadas[(total_saidas - 260):(num_amostras - 130)]

pesos_camada_escondida, pesos_camada_saida, erros, erros_validacao = pc.treinamento_validacao(x_train, y_train, taxa_aprendizado,
                                                                                              epocas, num_neuronios_ocultos, x_valid, y_valid)

# pesos_camada_escondida, pesos_camada_saida, erros = pc.treinamento(letras_treinamento, saidas_treinamento, taxa_aprendizado,
#                                                                                               epocas, numero_neuronios)

x_test = entradas_brutas[(num_amostras - 130):]
y_test = saidas_desejadas[(total_saidas - 130):]

print(pc.testar_rede(x_test, y_test, pesos_camada_escondida, pesos_camada_saida))

plt.plot(erros, label="Erro")
plt.plot(erros_validacao, label="Erro Validação")
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático Médio (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
