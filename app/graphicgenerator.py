import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
train_data = pd.read_csv('data/train.csv')

# Substituir valores faltantes
imputer = SimpleImputer(strategy='mean')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])

train_data['Fare'] = imputer.fit_transform(train_data[['Fare']])
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

with open('label_encoder_sex.pkl', 'rb') as file_sex:
    label_encoder_sex = pickle.load(file_sex)

with open('label_encoder_embarked.pkl', 'rb') as file_embarked:
    label_encoder_embarked = pickle.load(file_embarked)

# Converter variáveis categóricas
label_encoder_sex = label_encoder_sex
label_encoder_embarked = label_encoder_embarked

train_data['Sex'] = label_encoder_sex.fit_transform(train_data['Sex'].astype(str))
train_data['Embarked'] = label_encoder_embarked.fit_transform(train_data['Embarked'].astype(str))

# Funções para criar gráficos
def plot_survival_by_class():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Survived', data=train_data, palette='viridis', ax=ax)
    ax.set_title('Sobrevivência por Classe')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Número de Passageiros')
    ax.legend(['Não Sobreviveu', 'Sobreviveu'])
    return fig

def plot_survival_by_sex():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Sex', hue='Survived', data=train_data, palette='coolwarm', ax=ax)
    ax.set_title('Sobrevivência por Sexo')
    ax.set_xlabel('Sexo (0=Homem, 1=Mulher)')
    ax.set_ylabel('Número de Passageiros')
    ax.legend(['Não Sobreviveu', 'Sobreviveu'])
    return fig

def plot_class_vs_embarked():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Embarked', data=train_data, palette='plasma', ax=ax)
    ax.set_title('Distribuição de Passageiros por Classe e Local de Embarque')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Número de Passageiros')
    ax.legend(['C', 'Q', 'S'], title='Embarque')
    return fig
