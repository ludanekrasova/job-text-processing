# streamlit run Обработчик_вакансий.py

import streamlit as st
import pandas as pd
import json

import difflib

st.set_page_config(
    page_title="Разработка системы решения по обработке вакансий",
    page_icon="📋", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None,'Report a bug': None,'About': None})

hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

############################################################################
# Поиск по запросу соискателя
############################################################################


def get_similarity(find, df, n=5):
    # find - запрос соискателя
    # data - таблица с исправленными моделью вакансиями. Бот не будет их считать
    # n - количество выводимых вакансий. закодируй, чтобы красиво было 3-5 штук
    
    #возвращает отфильтвованный датафрейм подходящих вакансий
    data = df.copy()
    idxs = data.index.tolist()
    diffs=[]
    str_find = str(find.lower()) 
    for idx in idxs:
        resp = str(data['responsibilities(Должностные обязанности)'][idx])
        req = str(data['requirements(Требования к соискателю)'][idx])
        term = str(data['terms(Условия)'][idx])
        str_data = resp + " " +  req + " " +  term
        diff = difflib.SequenceMatcher(lambda x: x == " ", str_find, str_data).ratio() 
        diffs.append(round(diff, 3))
        
    data['similarity']=diffs
    #print(data)
    data.sort_values(by='similarity', ascending=False, inplace=True)
   
    # data.head(n)
    return data[0:n]

#загружаем датасет - результат модели (будет файл)
data_itog = pd.read_excel('Решение.xlsx', index_col=0)
# строка, которую вводит соискатель


############################################################################
# вывод результатов
############################################################################

with st.form('responsibilities', clear_on_submit=False):
    field = st.text_input('Ввод текста')
    # Every form must have a submit button.
    submitted = st.form_submit_button("Поиск")
    if submitted:
        find = field
        filter_data = get_similarity(find, data_itog)
        idxs = filter_data.index.tolist()
        for idx in idxs:
            result = f'''
**Вакансия:** {filter_data['name(название)'][idx]}\n
**Должностные обязанности:** {filter_data['name(название)'][idx]}\n
**Требования к соискателю:** {filter_data['requirements(Требования к соискателю)'][idx]} \n
**Условия работы:** {filter_data['terms(Условия)'][idx]} '''
            st.write(result)
            st.markdown("<hr>", unsafe_allow_html=True)