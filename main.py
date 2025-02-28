from aiogram import Bot, Dispatcher, F, Router, html, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from custom_pipeline import TfidfEmbeddingVectorizer
from parse import parse
from Model import get_propability
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import logging
TOKEN = '7612714979:AAEurD47dzd9rLE3eWFhjCVjqb2SHNluuhY'
bot = Bot(token=TOKEN)
dp = Dispatcher()
form_router = Router()
class UserSteps(StatesGroup):
    wait_date = State()
    wait_themes = State()


@form_router.message(CommandStart())
async def start(message: types.Message, state:FSMContext):
    await state.set_state(UserSteps.wait_date)
    await message.answer('Привет! введите дату в формате "дд.мм.гггг", в течении которой мы поищем статьи на ваши темы')

@form_router.message(UserSteps.wait_date)
async def get_pages(message: types.Message, state: FSMContext):
    try:
        await message.answer('Подождите, обрабатываем запрос...')
        date = (message.text).split('.')
    except:
        await message.answer('Возникла ошибка, введите дату в формате "дд.мм.гггг", в течении которой мы поищем статьи на ваши темы')

    try:
        global content
        content = parse(date)
        await state.set_state(UserSteps.wait_themes)
        await message.answer('Отлично, теперь отправьте пожалуйста темы, на которые вы хотите найти статьи: "Машинное обучение", "MLops"...')
    except Exception as e:
        print(e)
        await message.answer('Возникла ошибка, введите дату в формате "дд.мм.гггг", в течении которой мы поищем статьи на ваши темы')

@form_router.message(UserSteps.wait_themes)
async def get_top_articles(message: types.Message, state: FSMContext):
    try:
        await message.answer('Снова обрабатываем..')
        themes = (message.text).split(',')
    except:
        await message.answer(
            'Возникла ошибка, введите темы через запятую')

    try:
        print(content, themes)
        get_propability(content, themes)
    except:
        await message.answer(
            'Возникла ошибка, введите темы через запятую')




async def main():
    dp = Dispatcher()

    dp.include_router(form_router)
    await dp.start_polling((bot))

if __name__ == "__main__":
    asyncio.run(main())