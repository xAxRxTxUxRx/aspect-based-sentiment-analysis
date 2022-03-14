import preprocess_input
import final_model
import pandas as pd
import tensorflow as tf

cluster_map_restaurant = {0: 'Food', 1: 'Food', 2: 'Ambience', 3: 'Ambience',
                          4: 'Miscellaneous', 5: 'Food', 6: 'Food', 7: 'Miscellaneous', 8: 'Staff',
                          9: 'Price', 10: 'Food', 11: 'Staff',
                          12: 'Miscellaneous', 13: 'Ambience'}

cluster_map_beer = {0: 'overall', 1: 'taste+smell', 2: 'overall', 3: 'taste+smell',
                    4: 'None', 5: 'taste+smell', 6: 'feel', 7: 'None', 8: 'taste+smell',
                    9: 'None', 10: 'look', 11: 'taste+smell',
                    12: 'None', 13: 'None'}

#### Take final models
ABAE_restaurant = final_model.ABAE_model('restaurant', cluster_map_restaurant)
ABAE_restaurant_model = ABAE_restaurant.get_model()

ABAE_beer = final_model.ABAE_model('beer', cluster_map_beer)
ABAE_beer_model = ABAE_beer.get_model()

sentiment_model = final_model.Sentiment_Model()


def input_output(domain='restaurant', review=None):
    sentiment = 0
    if review is None:
        sent_input = input_field.get('1.0', 'end-1c')
        if sent_input:
            sentiment = sentiment_model.predict(sent_input)
    else:
        sent_input = review
        sentiment = sentiment_model.predict(sent_input)
    if check_button_value.get():
        sent_input = preprocess_input.single_input(sent_input, ABAE_restaurant.vocab, ABAE_restaurant.args['maxlen'])
        aspect_probs = ABAE_restaurant_model(sent_input)[0]
        aspect, prob = ABAE_restaurant.predict(aspect_probs)
    else:
        sent_input = preprocess_input.single_input(sent_input, ABAE_beer.vocab, ABAE_beer.args['maxlen'])
        aspect_probs = ABAE_beer_model(sent_input)[0]
        aspect, prob = ABAE_beer.predict(aspect_probs)
    output_label['text'] = f'Aspect: {aspect}. Sentiment: {sentiment}.'
    return aspect, sentiment


def input_file(source='', domain='restaurant', file=None):
    if file is None:
        file = open(source).readlines()
    d = {}
    if check_button_value.get():
        d = {'Food': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'Ambience': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'Staff': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'Price': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'Miscellaneous': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
        source = '../output/restaurant/analysed_file.xlsx'
    else:
        d = {'taste+smell': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'overall': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'look': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'feel': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             'None': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
        source = '../output/beer/analysed_file.xlsx'
        domain = 'beer'
    for line in file:
        aspect, sentiment = input_output(domain, line)
        d[aspect][sentiment] += 1
    output = pd.DataFrame({'Aspect': [key for key in d], '1': [d[key][1] for key in d],
                           '2': [d[key][2] for key in d], '3': [d[key][3] for key in d],
                           '4': [d[key][4] for key in d], '5': [d[key][5] for key in d]})
    output.to_excel(source)


#### GUI
from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

def select_files():
    text_filed4['text'] = 'Состояние анализа: в процессе.'
    filetypes = (
        ('text files', '*.txt'),
        ('All files', '*.*')
    )

    f = fd.askopenfile(filetypes=filetypes)
    f = f.readlines()
    input_file(file=f)
    text_filed4['text'] = 'Состояние анализа: готово.'


def download_file():
    if check_button_value:
        domain = 'restaurant'
    else:
        domain = 'beer'

    try:
        f = pd.read_excel(f'../output/{domain}/analysed_file.xlsx')
    except Exception:
        f = None

    source = fd.asksaveasfilename()
    f.to_excel(source)


bg_color = '#E5FFFC'
second_color = 'white'
root = Tk()

root.title('Aspect Based Sentiment Analysis'.upper())
root.configure(background=bg_color)

width = 1440
height = 720
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (width / 2))
y = int((screen_height / 2) - (height / 2))

root.geometry(f'{width}x{height}+{x}+{y}')


#### Главный канвас
canvas = Canvas(width=650, height=500, bd=0, highlightthickness=0)
canvas.configure(background=bg_color)
canvas.grid(padx=385, pady=80, columnspan=4, rowspan=4)

text_filed1 = Label(canvas, text='Введите отзыв', font='Inter 12')
text_filed1.configure(background=bg_color)
text_filed1.grid(sticky='w', row=0)

check_button_value = BooleanVar()
check_button = Checkbutton(canvas, text='Ресторан', font='Inter 12', onvalue=True, offvalue=False,
                           variable=check_button_value)
check_button.select()
check_button.configure(background=bg_color)
check_button.grid(sticky='e', row=0)

input_field = Text(canvas, font='Inter 18', width=50, height=2)
input_field.configure(background=second_color)
input_field.grid(pady=5)

output_label = Label(canvas, text='Аспект: ... Сентимент: ...', font='Inter 12')
output_label.configure(background=bg_color)
output_label.grid(pady=3, sticky='w')

input_button = Button(canvas, text='ТЕСТ', foreground='black', command=input_output, font='Inter 16', relief='flat',
                      width=13, height=1)
input_button.configure(background=second_color)
input_button.grid(pady=20)

title = 'Для анализа файших данных, нужно предоставить текстовый файл,\nсодержаший отзыв на каждой строке. На выходе вы получите файл,\nсодержищий информацию о каждом аспекте\n(кол-во отзывов и их сентимент).'
text_filed2 = Label(canvas, text=title, font='Inter 14', justify='left')
text_filed2.configure(background=bg_color)
text_filed2.grid(pady=30, sticky='w', column=0, columnspan=3)

text_filed3 = Label(canvas, text='Выбирите файл', justify='center', font='Inter 16')
text_filed3.configure(background=bg_color)
text_filed3.grid(pady=10)

open_file_button = Button(canvas, text='ОТКРЫТЬ ФАЙЛ', foreground='black', font='Inter 14', relief='flat', width=15,
                          height=1, command=select_files)
open_file_button.configure(background=second_color)
open_file_button.grid(pady=20, row=6, sticky='w')

text_filed4 = Label(canvas, text='Состояние анализа: ...', justify='center', font='Inter 12')
text_filed4.configure(background=bg_color)
text_filed4.grid(pady=5, sticky='w')

get_file_button = Button(canvas, text='ИТОГОВЫЙ ФАЙЛ', foreground='black', font='Inter 14', relief='flat', width=15,
                         height=1, command=download_file)
get_file_button.configure(background=second_color)
get_file_button.grid(pady=20, row=6, sticky='e')

root.mainloop()
