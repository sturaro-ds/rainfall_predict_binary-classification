# Modelos de Classificação Binária 
## Previsão de Chuvas | Base de competição Kaggle

<img src="https://img.freepik.com/fotos-premium/pato-bonito-com-guarda-chuva-na-chuva-ilustracao-de-icone-de-desenho-animado_1029476-412317.jpg" alt="Pato com guarda-chuva na chuva" width="400"/>

## Visão Geral

O objetivo deste projeto é prever a probabilidade de chuva em cada dia do dataset. A previsão é baseada em dados históricos meteorológicos, a base foi coletada do Kaggle conforme link a seguir: https://www.kaggle.com/competitions/playground-series-s5e3/overview

## Modelos Utilizados

Foram utilizados os seguintes modelos:
	•	Logistic Regression
	•	Random Forest
	•	Gradient Boosting
	•	SVC
	•	XGBoost

Cada modelo foi avaliado por métricas como AUC, precisão e RMSE, e o melhor modelo foi selecionado com base no ROC AUC.

## Etapas
	1.	Pré-processamento de Dados:
	•	Remoção de colunas desnecessárias e tratamento de valores ausentes.
	•	Conversão da variável alvo (‘rainfall’) para formato categórico.
	2.	Análise Exploratória de Dados:
	•	Visualização de distribuições das variáveis com gráficos de KDE, histogramas e boxplots.
	•	Normalização das variáveis com z-scores.
	3.	Treinamento dos Modelos:
	•	Treinamento do modelo base (Logistic Regression) e ajuste de hiperparâmetros com RandomizedSearchCV.
	4.	Avaliação:
	•	Geração de curvas ROC e relatórios de classificação para comparar o desempenho dos modelos.
	5.	Previsão:
	•	O melhor modelo foi utilizado para gerar as previsões e realizar a submissão.

## Instalação

Clone este repositório e execute os scripts:

git clone https://github.com/sturaro-ds/rainfall_predict_binary-classification.git
cd rainfall_predict_binary-classification

## Dependências

Instale as bibliotecas necessárias com o comando:

pip install -r requirements.txt

Observação: As explicações detalhadas no script estão em inglês, pois o projeto está sendo publicado na plataforma Kaggle.

## Licença

Este projeto está licenciado sob a Licença MIT.
