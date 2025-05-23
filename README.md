# EP1 IA

## Ambiente virtual python

```sh
python -m venv .venv
```

### Linux

```sh
source .venv/bin/activate
```

### Windows

```sh
.\.venv\Scripts\Activate
```

## Instalar as dependências

```sh
pip install -r requirements.txt
```


## Como Rodar o Código

```sh
usage: main.py [-h] -x ENTRADA -y SAIDA [-m {simples,validacao,kfold}] [-n NEURONIOS] [-t TAXA] [-e EPOCAS] [-f FOLDS] [-p {True,False}] [-a {sigmoide,tanh}] [--embaralhar]
               [--seed SEED]

Treinamento de rede neural para reconhecimento de letras.

options:
  -h, --help            show this help message and exit
  -x, --entrada ENTRADA
                        Caminho para o arquivo .npy com as entradas (X)
  -y, --saida SAIDA     Caminho para o arquivo .npy com as saídas (Y)
  -m, --modo {simples,validacao,kfold}
                        Modo de treinamento
  -n, --neuronios NEURONIOS
                        Número de neurônios na camada oculta
  -t, --taxa TAXA       Taxa de aprendizado
  -e, --epocas EPOCAS   Número de épocas de treinamento
  -f, --folds FOLDS     Número de folds (apenas se modo for kfold).
  -p, --plot {True,False}
                        Mostrar gráfico do erro quadrático médio
  -a, --ativacao {sigmoide,tanh}
                        Função de ativação usada na rede neural
  --embaralhar          Se definido, embaralha os dados antes do treinamento.
  --seed SEED           Seed para controlar a aleatoriedade (embaralhamento, inicialização dos pesos etc.)
```
