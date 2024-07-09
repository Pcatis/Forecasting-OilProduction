import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor

# Carrega os dados do arquivo Excel
dados_excel = pd.read_excel('D:\Desktop\Task and Research\JOB_ANALYST\JOB1_ABPIP_Aliquot\dadosf.xlsx', sheet_name='ana_gas')

# Cria um DataFrame vazio para armazenar as previsões
previsoes = pd.DataFrame(columns=['Empresa', 'Tempo', 'Previsão'])

# Cria um DataFrame vazio para armazenar as informações dos melhores modelos
modelos_best = pd.DataFrame(columns=['Empresa', 'Melhor Modelo', 'Erro Absoluto'])

# Cria um DataFrame vazio para armazenar as informações dos modelos e seus erros
modelos_erros = pd.DataFrame(columns=['Empresa', 'Método', 'Erro Absoluto'])

# Loop pelas empresas
for empresa in dados_excel.columns[1:]:
    erros_absolutos = []

    # Obtém os dados de tempo e produção para a empresa atual
    tempo = dados_excel['tempo(mes)']
    producao = dados_excel[empresa]

    # Divide os dados em treinamento (70%) e teste (30%)
    tempo_train, tempo_test, producao_train, producao_test = train_test_split(tempo, producao, test_size=0.3)

    # Regressão Linear
    modelo_linear = LinearRegression()
    modelo_linear.fit(tempo_train.values.reshape(-1, 1), producao_train)
    predicao_linear = modelo_linear.predict(tempo_test.values.reshape(-1, 1))
    erro_absoluto_linear = np.mean((abs(producao_test - predicao_linear)/producao_test))*100
    erros_absolutos.append(('Regressão Linear', erro_absoluto_linear))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'Regressão Linear', 'Erro Absoluto': erro_absoluto_linear}, ignore_index=True)

    # Regressão Polinomial (grau 2)
    modelo_polinomial = LinearRegression()
    polinomial_features = PolynomialFeatures(degree=2)
    tempo_poly_train = polinomial_features.fit_transform(tempo_train.values.reshape(-1, 1))
    modelo_polinomial.fit(tempo_poly_train, producao_train)
    tempo_poly_test = polinomial_features.transform(tempo_test.values.reshape(-1, 1))
    predicao_polinomial = modelo_polinomial.predict(tempo_poly_test)
    erro_absoluto_polinomial = np.mean((abs(producao_test - predicao_polinomial)/producao_test))*100
    erros_absolutos.append(('Regressão Polinomial', erro_absoluto_polinomial))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'Regressão Polinomial', 'Erro Absoluto': erro_absoluto_polinomial}, ignore_index=True)

    # Regressão Exponencial (SVR)
    modelo_exponencial = SVR()
    modelo_exponencial.fit(tempo_train.values.reshape(-1, 1), producao_train)
    predicao_exponencial = modelo_exponencial.predict(tempo_test.values.reshape(-1, 1))
    erro_absoluto_exponencial = np.mean((abs(producao_test - predicao_exponencial)/producao_test))*100
    erros_absolutos.append(('Regressão Exponencial', erro_absoluto_exponencial))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'Regressão Exponencial', 'Erro Absoluto': erro_absoluto_exponencial}, ignore_index=True)

    # K-Nearest Neighbors (KNN)
    modelo_knn = KNeighborsRegressor()
    modelo_knn.fit(tempo_train.values.reshape(-1, 1), producao_train)
    predicao_knn = modelo_knn.predict(tempo_test.values.reshape(-1, 1))
    erro_absoluto_knn = np.mean((abs(producao_test - predicao_knn)/producao_test))*100
    erros_absolutos.append(('K-Nearest Neighbors', erro_absoluto_knn))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'K-Nearest Neighbors', 'Erro Absoluto': erro_absoluto_knn}, ignore_index=True)

    # Support Vector Regression (SVR)
    modelo_svr = SVR()
    modelo_svr.fit(tempo_train.values.reshape(-1, 1), producao_train)
    predicao_svr = modelo_svr.predict(tempo_test.values.reshape(-1, 1))
    erro_absoluto_svr = np.mean((abs(producao_test - predicao_svr)/producao_test))*100
    erros_absolutos.append(('Support Vector Regression', erro_absoluto_svr))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'Support Vector Regression', 'Erro Absoluto': erro_absoluto_svr}, ignore_index=True)

    # Média Móvel Simples
    janela = 3
    media_movel_simples = producao.rolling(window=janela).mean().iloc[-1]
    erro_absoluto_media_movel_simples = np.mean((abs(producao_test.iloc[-1] - media_movel_simples)/producao_test.iloc[-1]))*100
    erros_absolutos.append(('Média Móvel Simples', erro_absoluto_media_movel_simples))
    modelos_erros = modelos_erros._append(
        {'Empresa': empresa, 'Método': 'Média Móvel Simples', 'Erro Absoluto': erro_absoluto_media_movel_simples},
        ignore_index=True)

    # Média Móvel Exponencial
    media_movel_exponencial = producao.ewm(span=janela, adjust=False).mean().iloc[-1]
    erro_absoluto_media_movel_exponencial = np.mean((abs(producao_test.iloc[-1] - media_movel_exponencial)/producao_test.iloc[-1]))*100
    erros_absolutos.append(('Média Móvel Exponencial', erro_absoluto_media_movel_exponencial))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'Média Móvel Exponencial',
                                          'Erro Absoluto': erro_absoluto_media_movel_exponencial}, ignore_index=True)

    # ARIMA
    modelo_arima = ARIMA(producao_train, order=(1, 0, 0))
    modelo_arima_fit = modelo_arima.fit()
    predicao_arima = modelo_arima_fit.predict(start=len(producao_train), end=len(producao_train) + len(producao_test) - 1)
    erro_absoluto_arima = np.mean((abs(producao_test - predicao_arima)/producao_test))*100
    erros_absolutos.append(('ARIMA', erro_absoluto_arima))
    modelos_erros = modelos_erros._append({'Empresa': empresa, 'Método': 'ARIMA', 'Erro Absoluto': erro_absoluto_arima},
                                          ignore_index=True)

    # Redes Neurais (MLPRegressor)
    modelo_rede_neural = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)
    modelo_rede_neural.fit(tempo_train.values.reshape(-1, 1), producao_train)
    predicao_rede_neural = modelo_rede_neural.predict(tempo_test.values.reshape(-1, 1))
    erro_absoluto_rede_neural = np.mean((abs(producao_test - predicao_rede_neural)/producao_test))*100
    erros_absolutos.append(('Redes Neurais', erro_absoluto_rede_neural))
    modelos_erros = modelos_erros._append(
        {'Empresa': empresa, 'Método': 'Redes Neurais', 'Erro Absoluto': erro_absoluto_rede_neural},
        ignore_index=True)

    # Seleciona o método com menor erro absoluto
    melhor_metodo = min(erros_absolutos, key=lambda x: x[1])

    # Faz a previsão para os próximos 60 meses usando o melhor método
    if melhor_metodo[0] == 'Regressão Linear':
        predicao_proximos_meses = modelo_linear.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'Regressão Polinomial':
        predicao_proximos_meses = modelo_polinomial.predict(
            polinomial_features.transform(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1)))
    elif melhor_metodo[0] == 'Regressão Exponencial':
        predicao_proximos_meses = modelo_exponencial.predict(
            np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'K-Nearest Neighbors':
        predicao_proximos_meses = modelo_knn.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'Support Vector Regression':
        predicao_proximos_meses = modelo_svr.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'Média Móvel Simples':
        predicao_proximos_meses = np.repeat(media_movel_simples, 60)
    elif melhor_metodo[0] == 'Média Móvel Exponencial':
        predicao_proximos_meses = np.repeat(media_movel_exponencial, 60)
    elif melhor_metodo[0] == 'ARIMA':
        modelo_arima = ARIMA(producao, order=(1, 0, 0))
        modelo_arima_fit = modelo_arima.fit()
        predicao_proximos_meses = modelo_arima_fit.predict(start=len(producao), end=len(producao) + 59)
    elif melhor_metodo[0] == 'Redes Neurais':
        predicao_proximos_meses = modelo_rede_neural.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))

    # Adiciona as previsões ao DataFrame
    previsoes_empresa = pd.DataFrame({'Empresa': empresa, 'Tempo': np.arange(max(tempo) + 1, max(tempo) + 61),
                                          'Previsão': predicao_proximos_meses})

    previsoes = previsoes._append(previsoes_empresa, ignore_index=True)

    # Adiciona as informações do melhor modelo ao DataFrame
    modelos_best = modelos_best._append({'Empresa': empresa, 'Melhor Modelo': melhor_metodo[0],
                                            'Erro Absoluto': melhor_metodo[1]}, ignore_index=True)

# Exporta o relatório de previsões para um arquivo Excel
previsoes.to_excel('previsoesgasRT1.xlsx', index=False)

# Exporta o relatório de melhores modelos para um arquivo Excel
modelos_best.to_excel('melhores_modelosgasRT1.xlsx', index=False)

# Exporta o relatório de modelos e erros para um arquivo Excel
modelos_erros.to_excel('modelos_errosgasRT1.xlsx', index=False)






'''
    # Faz a previsão para os próximos 60 meses usando o melhor método
    if melhor_metodo[0] == 'Regressão Linear':
        predicao_proximos_meses = modelo_linear.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'Regressão Polinomial':
        predicao_proximos_meses = modelo_polinomial.predict(
            polinomial_features.transform(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1)))
    elif melhor_metodo[0] == 'Regressão Exponencial':
        predicao_proximos_meses = modelo_exponencial.predict(
            np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'K-Nearest Neighbors':
        predicao_proximos_meses = modelo_knn.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))
    elif melhor_metodo[0] == 'Support Vector Regression':
        predicao_proximos_meses = modelo_svr.predict(np.arange(max(tempo) + 1, max(tempo) + 61).reshape(-1, 1))

    # Adiciona as previsões ao DataFrame
    previsoes_empresa = pd.DataFrame({'Empresa': empresa, 'Tempo': np.arange(max(tempo) + 1, max(tempo) + 61),
                                      'Previsão': predicao_proximos_meses})
    previsoes = previsoes._append(previsoes_empresa, ignore_index=True)

    # Adiciona as informações do melhor modelo ao DataFrame
    modelos_best = modelos_best._append({'Empresa': empresa, 'Melhor Modelo': melhor_metodo[0],
                                         'Erro Absoluto': melhor_metodo[1]}, ignore_index=True)

# Exporta o relatório de previsões para um arquivo Excel
previsoes.to_excel('previsoesgasT.xlsx', index=False)

# Exporta o relatório de melhores modelos para um arquivo Excel
modelos_best.to_excel('modelosbestgasT.xlsx', index=False)

# Exporta o relatório de modelos e erros para um arquivo Excel
modelos_erros.to_excel('modelos_errosgasT.xlsx', index=False)
'''





