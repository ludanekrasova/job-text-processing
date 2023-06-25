# streamlit run Обработчик_вакансий.py

import telebot

import pandas as pd
import json

import difflib


bot = telebot.TeleBot('6292922980:AAF5hSHEGe13pbHI-yrjrDCykPP7msKly_4')

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



@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    # строка, которую вводит соискатель
    find = message.text
    filter_data = get_similarity(find, data_itog)
    idxs = filter_data.index.tolist()
    for idx in idxs:
        result = f'''
*Вакансия:* {filter_data['name(название)'][idx]}
*Должностные обязанности:* {filter_data['name(название)'][idx]}
*Требования к соискателю:* {filter_data['requirements(Требования к соискателю)'][idx]} 
*Условия работы:* {filter_data['terms(Условия)'][idx]} '''
        bot.send_message(message.from_user.id, result, parse_mode="Markdown")

bot.polling(none_stop=True, interval=0)