import re
import time
import datetime
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from dataclasses import dataclass
import numpy as np
from custom_pipeline import TfidfEmbeddingVectorizer
SLEEP = 3
DEPTH = 100
URL = 'https://lenta.ru/parts/news/'


@dataclass
class Article:

    url: str = None
    title: str = None
    subtitle: str = None
    content: str = None
    datetime: str = None
    tag: str = None


# Настраиваем вебдрайвер
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('headless')
chrome_options.add_argument('no-sandbox')
chrome_options.add_argument('disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

from inspect import modulesbyfile
def get_pages(date):

    """Load and scroll pages"""
    items = []
    year = int(date[2])
    mounth = int(date[1])
    day =  int(date[0])
  # сначала скрлим в низ, потом нажимаем на кнопку прогрузки публикации
    if mounth < 10:
      mounth = f'0{mounth}'
    if day < 10:
        day = f'0{day}'
    for page in range(1, 2):
        try:
          driver.get(f'https://lenta.ru/{year}/{mounth}/{day}/page/{page}/')
          driver.execute_script(f'window.scrollTo(0, document.body.scrollHeight - 1200)')
          driver.execute_script("document.getElementsByClassName('loadmore')[0].click()")
          time.sleep(1)
          html = driver.page_source
          soup = BeautifulSoup(html, "html.parser")
          scope = soup.find('ul', {'class' : 'archive-page__container'})
          items_new = scope.find_all('li', {'class': 'archive-page__item _news'})
          items_to_append = [x for x in items_new if x not in items]
          items+=items_to_append
        except:
          print(f'страничка не прогрузилась')
          pass


    #возвращаем сборник статей
    return items

def parse_page(page):

    # создаем объект страницы-публикации
    article = Article()

    # загружаем страницу
    article.url = page.find('a', {'class': 'card-full-news _archive'})['href']
    base_url = 'https://lenta.ru'
    article.url = base_url + article.url

    # подгружаем страницу
    driver.get(article.url)
    time.sleep(SLEEP)
    html = driver.page_source

    # сохраняем ее адрес
    source = article.url[8: article.url.find('.')]

    # варим суп страницы
    soup = BeautifulSoup(html, "html.parser")
    obj = soup.find('div', {'class': 'topic-page__container'})

    # название статьи
    title = obj.find('span', {'class': 'topic-body__title'})
    title_2 = obj.find('h1', {'class': 'topic-body__title'})

    if title:
        article.title = title.text
    else:
        article.title = title_2.text if title_2 else ''

    # краткое описании статьи (если есть)
    subtitle = obj.find('div', {'class': 'topic-body__title-yandex'})
    article.subtitle = subtitle.text if subtitle else ''

    # содержание статьи
    article.content = obj.find('div', {'class': 'topic-body__content'}).text

    # тема статьи
    article.tag = obj.find('a', {'class': 'topic-header__item topic-header__rubric'}).text

    # время статьи
    article.datetime = obj.find('a', {'class': 'topic-header__item topic-header__time'}).text

    return article

def parse(date):
    pages = get_pages(date)

    data = []
    for page in tqdm(pages):
        try:
            res = parse_page(page)
            data.append(res)
        except:
            pass
    driver.close()

    df = pd.DataFrame(data=data)
    content = df['content'].tolist()
    return content



