import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import StudentData
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, cohen_kappa_score, f1_score, recall_score, roc_curve, auc, precision_score
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração do banco de dados
db_host = 'localhost'  # substitua pelo valor correto
db_port = '3306'  # substitua pelo valor correto
db_name = 'evasionwatch'  # substitua pelo valor correto
db_username = 'root'  # substitua pelo valor correto
db_password = 'QweBHU*'  # substitua pelo valor correto

db_url = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()

# Funções de IA:

# Função para plotar as matrizes de confusão
def plotarMatriz(cm):
  # Define as classes
  classes = ["Não Evadido", "Evadido"]

  # Plot a matriz de confusão com Seaborn
  sns.set(font_scale=1.0)
  sns.heatmap(cm, annot=True, annot_kws={"size": 12}, cmap="Blues", fmt="g", xticklabels=classes, yticklabels=classes)

  # Configurações adicionais
  plt.xlabel('Predito')
  plt.ylabel('Verdadeiro')
  plt.show()

# Função para plotar curva ROC/AUC
def plotarCurvaROC_AUC(X_test, y_test, model, title):
  # Calcular as probabilidades de classe positiva para as amostras de teste
  y_prob = model.predict_proba(X_test)[:, 1]

  # Extrair a probabilidade positiva de cada amostra (classe 1) e as classes verdadeiras (y_true)
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)

  # Calcular a área sob a curva ROC
  roc_auc = auc(fpr, tpr)

  # Plotar a curva ROC
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Taxa de Falso Positivo')
  plt.ylabel('Taxa de Verdadeiro Positivo')
  plt.title(title)
  plt.legend(loc="lower right")
  plt.show()

def trainXGB(X, y, params):
  # Define o modelo XGBoost
  model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, **params)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # # Define a validação cruzada com 5 folds
  # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

  # # Realiza a validação cruzada e calcula as métricas
  # y_true, y_pred = [], []
  # for train_idx, test_idx in kfold.split(X):
  #     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
  #     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
  #     model.fit(X_train, y_train)
  #     y_pred.extend(model.predict(X_test))
  #     y_true.extend(y_test)

  # Treina o modelo usando o conjunto de treino

  model.fit(X_train, y_train)

  # Faz previsões usando o conjunto de teste
  y_pred = model.predict(X_test)
  y_true = y_test
      
  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average='weighted')
  recall = recall_score(y_true, y_pred, average='weighted')
  precision = precision_score(y_true, y_pred)
  model_score = cross_val_score(model, X, y, cv=5)
  kappa = cohen_kappa_score(y_true, y_pred)

  # Exibe as métricas
  print(f'Acurácia: {acc:.4f}')
  print(f'F1-Score: {f1:.4f}')
  print(f'Recall: {recall:.4f}')
  print(f'Precision: {precision:.4f}')
  print(f'Cohen-Kappa: {kappa:.4f}')
  print(f'model_score: {model_score}')
  print(f'model_score: {model_score.mean()}')

  # Exibe as colunas e sua porcentagem de importancia
  print("--------")
  printFeatureImportances(model, X)

  # Calcula a matriz de confusão
  print("")
  cm = confusion_matrix(y_true, y_pred)
  plotarMatriz(cm)
  
  # Plota o gráfico de curva ROC/AUC
  print("")
  plotarCurvaROC_AUC(X_test, y_test, model, "Curva ROC do modelo XGBoost")
  
def trainDecisionTree(X, y, params):
  # Divide o dataset em conjunto de treino e conjunto de teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Cria o modelo de árvore de decisão
  model = tree.DecisionTreeClassifier(**params)

  # Treina o modelo usando o conjunto de treino
  model.fit(X_train, y_train)

  # Faz previsões usando o conjunto de teste
  y_pred = model.predict(X_test)

  # Avalia a acurácia do modelo usando o conjunto de teste
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  precision = precision_score(y_test, y_pred)
  kappa = cohen_kappa_score(y_test, y_pred)

  # Exibe as métricas
  print(f'Acurácia: {acc:.4f}')
  print(f'F1-Score: {f1:.4f}')
  print(f'Recall: {recall:.4f}')
  print(f'Precision: {precision:.4f}')
  print(f'Cohen-Kappa: {kappa:.4f}')

  cm = confusion_matrix(y_test, y_pred, normalize=None)
  plotarMatriz(cm)
  plotarCurvaROC_AUC(X_test, y_test, model, "Curva ROC do modelo DecisionTreeClassifier")

def gridSearchXGB(X, y, param_grid):
  # Define o modelo XGBoost
  model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

  # Define a validação cruzada com 5 folds
  kfold = KFold(n_splits=5, shuffle=True, random_state=42)

  # Realiza o grid search com os valores definidos acima
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='accuracy', verbose=3)

  # Executa o grid search
  grid_search.fit(X, y)

  # Exibe os melhores parâmetros e score
  print(f'Melhores parâmetros: {grid_search.best_params_}')
  print(f'Melhor score: {grid_search.best_score_:.4f}')

def printFeatureImportances(model, X):
  # Obtém a importância de cada coluna
  importance = model.feature_importances_

  # Cria um DataFrame com a importância de cada coluna e seu nome
  feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})

  # Ordena as colunas pela importância
  feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

  # Exibe o resultado
  print(feature_importance)

# Montando uma lista com os títulos das colunas no dataframe
colunas = ["descricao_modalidade_curso", "descricao_cota", "nivel_ensino",
           "descricao_situacao_matricula", "descricao_estado_civil", "descricao_turno", "descricao_naturalidade",
           "coeficiente_rendimento", "descricao_cor", "sexo", "descricao_curso", "percentual_frequencia",
           "descricao_tipo_escola_origem", "descricao_renda_per_capita"]

# Consulta para recuperar os registros da tabela
features = session.query(StudentData).all()

# Criando um dicionário com as colunas e os valores das features
data = {coluna: [getattr(f, coluna) for f in features] for coluna in colunas}

# Criando o DataFrame
qData = pd.DataFrame(data)

# Removendo alunos que optaram cancelar a matrícula, não categorizado como abandono/evasão
qData = qData[(qData['descricao_situacao_matricula'] != 'Cancelamento Voluntário') & (qData['descricao_situacao_matricula'] != 'Trancado')]

# Criando uma tabela target separando evadidos dos em situação regular
qData['target'] = np.where(qData['descricao_situacao_matricula'] == "Abandono/Evasão", 1, 0)

# Substituindo os textos(Strings) por valores a coluna de sexo
sexo_dict = {"M": 0, 
             "F": 1}
qData["sexo"] = qData["sexo"].map(sexo_dict).fillna(qData["sexo"])

qData.replace("", np.nan, inplace=True)

# Atribuindo "NÃO DECLARADO" as linhas da coluna estado civil em branco (NaN)
qData["descricao_estado_civil"].fillna("NÃO DECLARADO", inplace=True)

# Substituindo os textos(Strings) por valores a coluna de estado civil
estado_civil_dict = {"NÃO DECLARADO": 0,
                     "CASADO (A)": 1,
                     "DIVORCIADO (A)": 2,
                     "OUTROS": 3,
                     "SEPARADOS": 4,
                     "SOLTEIRO (A)": 5,
                     "UNIÃO ESTÁVEL": 6}

qData["descricao_estado_civil"] = qData["descricao_estado_civil"].map(estado_civil_dict).fillna(qData["descricao_estado_civil"])

# Substituindo os textos(Strings) por valores a coluna de cor
cor_dict = {"Não dispõe da informação": 0,
            "Não quis declarar cor/raça": 1,
            "Amarela": 2,
            "Branca": 3,
            "Indígena": 4,
            "Parda": 5,
            "Preta": 6}
qData["descricao_cor"] = qData["descricao_cor"].map(cor_dict).fillna(qData["descricao_cor"])

# Substituindo os textos(Strings) por valores a coluna de tipo de curso
course_type_dict = {"Concomitante": 0,
                    "Especialização": 1,
                    "Subseqüente": 2,
                    "Tecnólogo": 3}
qData["descricao_modalidade_curso"] = qData["descricao_modalidade_curso"].map(course_type_dict).fillna(qData["descricao_modalidade_curso"])

# Substituindo os textos(Strings) por valores a coluna de curso
curso_dict = {"ASSISTENTE ADMINISTRATIVO - PROEJA - SEE/PE": 0,
              "ESPECIALIZAÇÃO EM DESENVOLVIMENTO,INOVAÇÃO E TECNOLOGIAS EMERGENTES": 1,
              "ESPECIALIZAÇÃO EM GESTÃO E QUALIDADE EM TECNOLOGIA DA INFORMAÇÃO E COMUNICAÇÃO": 2,
              "TÉCNICO EM INFORMÁTICA PARA INTERNET - SUBSEQUENTE": 3,
              "TÉCNICO EM QUALIDADE - SUBSEQUENTE": 4,
              "TECNOLOGIA EM ANÁLISE E DESENVOLVIMENTO DE SISTEMAS": 5}
qData["descricao_curso"] = qData["descricao_curso"].map(curso_dict).fillna(qData["descricao_curso"])

# Substituindo os textos(Strings) por valores a coluna de turno
turno_dict = {"Matutino": 0,
              "Noturno": 1,
              "Vespertino": 2}
qData["descricao_turno"] = qData["descricao_turno"].map(turno_dict).fillna(qData["descricao_turno"])

# Substituindo os textos(Strings) por valores a coluna de tipo de cota
cota_dict = {"Não possui": 0,
             "Aluno de Escola Pública com renda <= 1,5 SM por pessoa": 1,
             "Aluno de Escola Pública com renda <= 1,5 SM por pessoa, autodeclarado preto, pardo ou indígena": 2,
             "Aluno de Escola Pública com renda > 1,5 SM por pessoa": 3,
             "Aluno de Escola Pública com renda > 1,5 SM por pessoa, autodeclarado preto, pardo ou indígena": 4,
             "Aluno de Escola Pública, com deficiência e com renda <= 1,5 SM por pessoa": 5,
             "Aluno de Escola Pública, com deficiência e com renda <= 1,5 SM por pessoa, autodeclarado preto, pardo ou indígena": 6,
             "Aluno de Escola Pública, com deficiência e com renda > 1,5 SM por pessoa": 7,
             "Alunos de Escola Pública, sem comprovação renda": 8,
             "Alunos de Escola Pública, sem comprovação renda, autodeclarado preto, pardo ou indígena": 9,
             "Ampla Concorrência": 10}
qData["descricao_cota"] = qData["descricao_cota"].map(cota_dict).fillna(qData["descricao_cota"])

# Preparando os dados para o treinamento do modelo

# Converte decimalSeparator para poder converter para float
qData['coeficiente_rendimento'] = qData['coeficiente_rendimento'].str.replace(',', '.').astype(float)

# Substituir strings vazias por NaN
qData['percentual_frequencia'] = qData['percentual_frequencia'].replace('', np.nan)

qData['percentual_frequencia'] = qData['percentual_frequencia'].str.replace(',', '.').astype(float)

# Atribuir valores as colunas de ESTADO
cidadeDict = {}
lastIndex = 0

for index, row in qData.iterrows():
    if(row['descricao_naturalidade'] not in cidadeDict):
      lastIndex = lastIndex + 1
      cidadeDict[row['descricao_naturalidade']] = lastIndex

    qData.loc[index, 'descricao_naturalidade'] = cidadeDict[row['descricao_naturalidade']]
qData['descricao_naturalidade'] = qData['descricao_naturalidade'].astype(int)    
    
# Atribuir valores as colunas de nivel_ensino
nvEnsinoDict = {}
lastIndex = 0

for index, row in qData.iterrows():
    if(row['nivel_ensino'] not in nvEnsinoDict):
      lastIndex = lastIndex + 1
      nvEnsinoDict[row['nivel_ensino']] = lastIndex

    qData.loc[index, 'nivel_ensino'] = nvEnsinoDict[row['nivel_ensino']]
qData['nivel_ensino'] = qData['nivel_ensino'].astype(int)

# Atribuir valores as colunas de tipo de escola origem
tipoEnsinoDict = {}
lastIndex = 0

for index, row in qData.iterrows():
    if(row['descricao_tipo_escola_origem'] not in tipoEnsinoDict):
      lastIndex = lastIndex + 1
      tipoEnsinoDict[row['descricao_tipo_escola_origem']] = lastIndex

    qData.loc[index, 'descricao_tipo_escola_origem'] = tipoEnsinoDict[row['descricao_tipo_escola_origem']]
qData['descricao_tipo_escola_origem'] = qData['descricao_tipo_escola_origem'].astype(int)


# Atribuir valores as colunas de descricao_renda_per_capita
rendaPerCapitaDict = {}
lastIndex = 0

for index, row in qData.iterrows():
    if(row['descricao_renda_per_capita'] not in rendaPerCapitaDict):
      lastIndex = lastIndex + 1
      rendaPerCapitaDict[row['descricao_renda_per_capita']] = lastIndex

    qData.loc[index, 'descricao_renda_per_capita'] = rendaPerCapitaDict[row['descricao_renda_per_capita']]
qData['descricao_renda_per_capita'] = qData['descricao_renda_per_capita'].astype(int)

# Iniciando separação dos dados para predição

# Preencher NaN com -1
qData['percentual_frequencia'] = qData['percentual_frequencia'].fillna(-1)

# Remove os elementos target
qData = qData.drop("descricao_situacao_matricula", axis=1)

# Monta o X e Y responsáveis pelo treinamento
X = qData.drop("target", axis=1)
y = qData["target"]

params = {'colsample_bytree': 1, 'gamma': 4, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1, "min_child_weight": 5}

trainXGB(X, y, params)