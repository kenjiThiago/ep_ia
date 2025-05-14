# EP1 IA

## Ambiente virtual python

```sh
python -m venv .venv
```

```sh
source .venv/bin/activate
```


## Instalar as dependências

```sh
pip install -r requirements.txt
```


## Como Rodar o Código

```sh
usage: main.py [-h] -m {simples,validacao,kfold} -n NEURONIOS -t TAXA -e EPOCAS [-f FOLDS] [-p {True,False}]

Treinamento de rede neural para reconhecimento de letras.

options:
  -h, --help            show this help message and exit
  -m, --modo {simples,validacao,kfold}
                        Modo de treinamento
  -n, --neuronios NEURONIOS
                        Número de neurônios na camada oculta
  -t, --taxa TAXA       Taxa de aprendizado
  -e, --epocas EPOCAS   Número de épocas de treinamento
  -f, --folds FOLDS     Número de folds (apenas se modo for kfold).
  -p, --plot {True,False}
                        Mostrar gráfico do erro quadrático médio
```
