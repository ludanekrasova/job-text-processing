# streamlit run Обработчик_вакансий.py

import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
#import xlsxwriter
import openpyxl


st.set_page_config(
    page_title="Разработка системы решения по обработке вакансий",
    page_icon="📋", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None,'Report a bug': None,'About': None})

hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

############################################################################
# работа с файлами
############################################################################

@st.cache_data
def load_model():
    '''Загрузка модели'''
    # загрузка моделей
    model.load_model('data/model.cbm')
    # загрузка векторизатора
    tfidf = joblib.load('data/tfidf.pkl')
    return model, tfidf


@st.cache_data
def load_data():
    '''Загрузка файла'''
    data = pd.read_excel('data/dataset.xlsx', engine='openpyxl').head(10)
    return data
def tfidf_featuring(tfidf, df):
    '''Преобразование текста в мешок слов'''
    X_tfidf = tfidf.transform(df['text'])
    feature_names = tfidf.get_feature_names_out()
    X_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=feature_names, index=df.index)
    return X_tfidf

def sentences_split(text):
    # разбивает текст на предложения
    try:
        sentences = re.compile(r'\;|\.|\n|\•|—|обязанности|требования|условия').split(text.lower())
        pattern = r'^[^а-яА-ЯёЁ]+|[^а-яА-ЯёЁ]+$'
        sentences = [re.sub(pattern, '', sen) for sen in sentences]
        return [sen for sen in sentences if len(sen) > 0]
    except:
        return []

def sentences_df(df, part=None):
    # датафрейм для извлечения, part сколько строк извдлекаем
    test_ = []
    idxs = df.index.tolist()
    for idx in idxs[0:part]:
        text = df['responsibilities(Должностные обязанности)'][idx]
        # print(sentences_split(text))
        test_.append(sentences_split(text))
    return test_

def sort_respons(sentences):
    requirements = []
    terms = []
    for idx in tqdm(range(0, len(sentences))):
        test_tfidf = tfidf_featuring(tfidf, pd.DataFrame({"text": sentences[idx]}))
        catc_proba = model.predict_proba(test_tfidf)
        temp_ = pd.DataFrame({"text": sentences[idx]})
        temp_['target'] = np.argmax(catc_proba, axis=1)
        temp_['proba'] = np.amax(catc_proba, axis=1)
        temp_['target'] = temp_['target'].replace({0: "Обязанности", 1: 'Требования',
                                                   2: 'Условия работы'}, regex=True)
        # пока без обязанностей
        # responsibilities = temp_[(temp_['target']=='Обязанности')&(temp_['proba']>=0.75)]['text'].tolist()
        requirements.append(temp_[(temp_['target'] == 'Требования') & (temp_['proba'] >= 0.75)]['text'].tolist())
        terms.append(temp_[(temp_['target'] == 'Условия работы') & (temp_['proba'] >= 0.75)]['text'].tolist())

    return requirements, terms

def print_list(text):
    # вывод списка построчно
    for i in text:
        st.text("- " + i.capitalize() + ";")


######################################################################################################
# Основной код
######################################################################################################

model = CatBoostClassifier(loss_function='MultiClass', random_state=42)
tfidf = TfidfVectorizer()
# загрузка классификатора и векторизатора
model, tfidf = load_model()

#загрузка файла с вакансиями
data = load_data()



# предсказание требований и условий
sentences = sentences_df(data)
requirements, terms = sort_respons(sentences)

data['requirements(Требования к соискателю)'] = requirements
data['terms(Условия)'] = terms

############################################################################
# вывод результатов
############################################################################

#result = result_default
counter_vac = data.shape[0] - 1
# стрелочки для вакансий
button_container = st.container()
with button_container:
    if 'count' not in st.session_state:
        st.session_state.count = 0
    col1, col2, col3 = st.columns(3)
    with col1:
        decrement = st.button('←')
        if decrement and st.session_state.count > 0:
            st.session_state.count -= 1  # Уменьшение индекса при нажатии стрелки "←"
    with col2:
        idx = st.session_state.count
        st.write('Вакансия = ', st.session_state.count)
    with col3:
        increment = st.button('→')
        if increment and st.session_state.count < counter_vac:
            st.session_state.count += 1  # Увеличение индекса при нажатии стрелки "→"

    # Получение значения ячейки по имени столбца и индексу строки
    cell_value = data.at[st.session_state.count, 'responsibilities(Должностные обязанности)']
    name_value = data.at[st.session_state.count, 'name(название)']
    requirements_value = data.at[st.session_state.count,'requirements(Требования к соискателю)']
    terms_value = data.at[st.session_state.count,'terms(Условия)']

    # Вывод значения ячейки

    st.markdown("<h4>Название вакансии:</h4>", unsafe_allow_html=True)
    st.write(name_value)

    st.markdown("<h4>Первоначальный текст:</h4>", unsafe_allow_html=True)
    #st.subheader("Сырой текст:")
    st.write(cell_value)

    st.markdown("<h4>Обработанные данные:</h4>", unsafe_allow_html=True)

    st.markdown("<h4>Требования:</h4>", unsafe_allow_html=True)
    # список требований
    print_list(requirements_value)

    st.markdown("<h4>Условия:</h4>", unsafe_allow_html=True)
    # список условий
    print_list(terms_value)

