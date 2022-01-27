# Bikes Sharing Predictor

## Introduzione

Bikes Sharing Predictor è il mio primo esperimento **Python** su **Machine Learning** realizzato a completamento del corso YouTube di [Piero Savastano](https://www.youtube.com/channel/UCD-HLhRV_4Z3sYGkgqAnIJw):
- [Progetto biciclette - esempio di machine learning in python](https://www.youtube.com/playlist?list=PLa-sizbCyh931pBRl4IOxiHXDIvjSzDh0)

Il progetto è stato realizzato con l'obiettivo di migliorare il modello di machine learning illustrato nel tutorial, 
prendendo spunto dalle nozioni apprese durante il corso e cercando di applicare alcune ulteriori tecniche:

- Cyclical Features
- Neural Networks
- Model Persistence

## Struttura del progetto

La cartella `data/` include un README con informazioni riguardanti il dataset e contiene il dataset originale del progetto con in più l'aggiunta di un dataset custom appositamente generato per l'addestramento del modello.

La cartella `model/dump/` contiene il modello `mlpregressor-20201011` addestrato in precedenza.

Il codice sorgente è costituito dai seguenti script python:
- `models/mplregressor`: Classe che implementa il *modello MLP Regressor* su cui si basa il progetto
- `dataset.py`: Classe utilizzata per la *generazione del dataset custom* a partire dal dataset master
- `main.py`: File utilizzato per *addestrare e testare il modello* di Maching Learning con il dataset

**NOTE**

1. Allo stato attuale il file main.py esegue il test del modello 'mlpregressor-20201011' precedentemente addestrato (con l'istruzione `model_train(model, dataset)` adesso commentata all'interno dello stesso file main.py).

2. La qualità del codice sorgente non è sicuramente delle migliori, e potrebbe essere notevolmente migliorata con un po' di refactoring, applicando alcuni design patterns e tenendo conto delle competenze professionali acquisite successivamente alla prima stesura del progetto (Ottobre 2020).

3. Ulteriori dettagli implementativi sono consultabili nei commenti di alcuni video YouTube, come ad esempio:
   - [Machine learning in python - 25 - Migliorare le predizioni [progetto biciclette]](https://www.youtube.com/watch?v=EYZquSyo2ws)

## Setup

Il progetto è stato realizzato inizialmente con PyCharm e poi riconvertito per essere eseguito direttamente da:
- **Visual Studio Code** con estensione [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

Per maggiori informazioni:
- [Visual Studio Code Remote Containers](https://code.visualstudio.com/docs/remote/containers)

Dopo aver aperto il progetto per la prima volta sarà necessario installare le librerie richieste:

```bash
pip install -r requirements.txt
```

Una volta installate le librerie necessarie sarà possibile provare il modello MLP Regressor realizzato:

```bash
python main.py
```

```console
Train: mean_absolute_error=25.123095235591116, r2_score=0.9498678974685386
```

Il suddetto output mostra i risultati ottenuti dal *modello MLP Regressor* utilizzato:
- MAE = 25.12
- R2 = 94.98%

[Mean Absolute Error (MAE)](https://c3.ai/glossary/data-science/mean-absolute-error/)
> In the context of machine learning, absolute error refers to the magnitude of difference between the prediction of an observation and the true value of that observation. MAE takes the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group. MAE can also be referred as L1 loss function.

[R-Squared (R2)](https://c3.ai/glossary/data-science/coefficient-of-discrimination-r-squared-r2/)
> The coefficient of discrimination, denoted 𝑅², is a commonly used performance metric for regression. It provides a measure of the proportion of the variance of a dependent variable that is explained by a regression model...

## Risorse utili:

- [Bike Sharing Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- [Feature Engineering - Handling Cyclical Features](http://blog.davidkaleko.com/feature-engineering-cyclical-features.html)
- [Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html)
- [Save and Load Machine Learning Models in Python with scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)
