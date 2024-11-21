import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Carregar o modelo treinado
with open('model/titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Carregar o escalador treinado
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Função para receber dados do usuário
def user_input_features():
    pclass = st.selectbox('Classe', [1, 2, 3])
    sex = st.selectbox('Sexo', ['male', 'female'])
    age = st.slider('Idade', 0, 100, 30)
    sibsp = st.slider('Número de irmãos/cônjuges', 0, 10, 0)
    parch = st.slider('Número de pais/filhos', 0, 10, 0)
    fare = st.number_input('Tarifa', 0.0, 500.0, 50.0)
    embarked = st.selectbox('Embarque', ['C', 'Q', 'S'])

    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Função para fazer a previsão
def predict_survival(input_data):
    # Mapeando as variáveis categóricas para numéricas
    input_data['Sex'] = input_data['Sex'].map({'male': 0, 'female': 1})
    input_data['Embarked'] = input_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Usar o mesmo conjunto de dados (com todas as colunas) que foi usado para treinar o scaler
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    input_data = input_data[features]
    
    # Normalizando os dados (aplicando o scaler da mesma maneira que foi feito durante o treino)
    input_data_scaled = scaler.transform(input_data) 
    
    # Realizando a previsão
    prediction = model.predict(input_data_scaled)
    print(input_data)
    return prediction

# Interface de usuário
st.title('Previsão de Sobrevivência Titanic')
st.write("Preencha os dados abaixo para prever a sobrevivência no Titanic")

user_data = user_input_features()

if st.button('Prever'):
    result = predict_survival(user_data)
    if result == 1:
        st.success("Sobreviveu")
    else:
        st.error("Não Sobreviveu")
