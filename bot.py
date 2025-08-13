import os
import json
import logging
import nest_asyncio
from aiogram.fsm.context import FSMContext  
from PIL import Image, ImageFile
import io
nest_asyncio.apply()
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import re
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.exceptions import AiogramError
import aiofiles
import html
from aiogram.enums import ParseMode
import sqlite3
from contextlib import closing
from datetime import datetime
from aiogram.types import BotCommand
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from typing import Union
from natasha import MorphVocab
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from natasha import Segmenter, MorphVocab, NewsMorphTagger, NewsEmbedding, Doc
from typing import Dict, List, Set
import re
from functools import lru_cache












# --- Настройка логов ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Явная загрузка .env ---
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("Не задан TELEGRAM_BOT_TOKEN в .env файле")

# --- Константы callback_data ---
CALLBACK_TALES = "tales"
CALLBACK_VOCABULARY = "vocabulary"
CALLBACK_GRAMMAR = "grammar"
CALLBACK_LEXICON = "lexicon"
CALLBACK_BACK_TO_MAIN = "back_to_main"
CALLBACK_BACK_TO_TALES = "back_to_tales"
CALLBACK_BACK_TO_VOCABULARY = "back_to_vocabulary"
CALLBACK_SHOW_STORY = "show_story_"
CALLBACK_SHOW_GRAMMAR = "show_grammar_"
CALLBACK_SHOW_LEXICON = "show_lexicon_"
CALLBACK_LANGUAGE_RU = "lang_ru_"
CALLBACK_LANGUAGE_KH = "lang_kh_"
CALLBACK_BACK_TO_LANGUAGE = "back_to_lang_"
CALLBACK_PLAY_AUDIO = "play_audio_"
CALLBACK_ALPHABET = "alphabet"
CALLBACK_ALPHABET_LETTERS = "alphabet_letters"
CALLBACK_ALPHABET_VOWELS = "alphabet_vowels"
CALLBACK_ALPHABET_CONSONANTS = "alphabet_consonants"
CALLBACK_TALES_PAGE_PREFIX = "tales_page_"
CALLBACK_TALES_PREV = "tales_prev"
CALLBACK_TALES_NEXT = "tales_next"
CALLBACK_ALPHABET_LETTERS_LIST = "alphabet_letters_list"
CALLBACK_ALPHABET_LETTER_DETAIL = "alphabet_letter_detail:"
CALLBACK_ALPHABET_DESCRIPTION = "alphabet_desc"
CALLBACK_VOWELS_DESCRIPTION = "vowels_desc"
CALLBACK_CONSONANTS_DESCRIPTION = "consonants_desc"
CALLBACK_SHOW_ILLUSTRATIONS = "show_illustrations_"
CALLBACK_PROGRESS = "show_progress"
CALLBACK_SHOW_CULTURE = "show_culture_"


















# --- Класс для работы с базой данных ---
class Database:
    def __init__(self, db_name: str = "user_progress.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        """Инициализация базы данных и создание таблиц"""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.cursor()
            # Таблица пользователей
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    registration_date TEXT
                )
            """)
            # Таблица прогресса по сказкам
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tale_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    tale_id INTEGER,
                    last_read_date TEXT,
                    read_count INTEGER DEFAULT 0,
                    completed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            # Таблица результатов тестов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    tale_id INTEGER,
                    question_id INTEGER,
                    is_correct BOOLEAN,
                    answer_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            conn.commit()

    def add_user(self, user: types.User):
        """Добавление нового пользователя в базу данных"""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO users 
                (user_id, username, first_name, last_name, registration_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user.id,
                    user.username,
                    user.first_name,
                    user.last_name,
                    datetime.now().isoformat()
                )
            )
            conn.commit()

    def update_tale_progress(self, user_id: int, tale_id: int) -> bool:
        """Обновление прогресса по сказке. Возвращает True, если запись была обновлена, False если создана новая"""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, read_count FROM tale_progress WHERE user_id = ? AND tale_id = ?",
                (user_id, tale_id)
            )
            record = cursor.fetchone()
            
            if record:
                read_count = record[1] + 1
                cursor.execute(
                    """
                    UPDATE tale_progress 
                    SET read_count = ?, last_read_date = ?
                    WHERE id = ?
                    """,
                    (read_count, datetime.now().isoformat(), record[0])
                )
                conn.commit()
                return True
            else:
                cursor.execute(
                    """
                    INSERT INTO tale_progress 
                    (user_id, tale_id, last_read_date, read_count)
                    VALUES (?, ?, ?, 1)
                    """,
                    (user_id, tale_id, datetime.now().isoformat())
                )
                conn.commit()
                return False

    def mark_tale_completed(self, user_id: int, tale_id: int):
        """Помечаем сказку как завершенную (пройден тест)"""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM tale_progress WHERE user_id = ? AND tale_id = ?",
                (user_id, tale_id)
            )
            record = cursor.fetchone()
            if record:
                cursor.execute(
                    """
                    UPDATE tale_progress 
                    SET completed = TRUE, last_read_date = ?
                    WHERE id = ?
                    """,
                    (datetime.now().isoformat(), record[0])
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO tale_progress 
                    (user_id, tale_id, last_read_date, completed)
                    VALUES (?, ?, ?, TRUE)
                    """,
                    (user_id, tale_id, datetime.now().isoformat())
                )
            conn.commit()

    def save_test_result(self, user_id: int, tale_id: int, question_id: int, is_correct: bool):
        """Сохранение результата ответа на вопрос теста"""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_results 
                (user_id, tale_id, question_id, is_correct, answer_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    tale_id,
                    question_id,
                    is_correct,
                    datetime.now().isoformat()
                )
            )
            conn.commit()

    def get_user_progress(self, user_id: int) -> dict:
        """Получение прогресса пользователя"""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT tale_id) as tales_read,
                    SUM(read_count) as total_reads,
                    COUNT(DISTINCT CASE WHEN completed THEN tale_id END) as tales_completed
                FROM tale_progress
                WHERE user_id = ?
            """, (user_id,))
            stats = cursor.fetchone()
            tales_read = stats[0] if stats and stats[0] is not None else 0
            total_reads = stats[1] if stats and stats[1] is not None else 0
            tales_completed = stats[2] if stats and stats[2] is not None else 0

            cursor.execute("""
                SELECT tale_id, last_read_date, read_count, completed
                FROM tale_progress
                WHERE user_id = ?
                ORDER BY last_read_date DESC
                LIMIT 5
            """, (user_id,))
            recent_tales = cursor.fetchall()

            return {
                "tales_read": tales_read,
                "total_reads": total_reads,
                "tales_completed": tales_completed,
                "recent_tales": recent_tales
            }

# --- Загрузка сказок из JSON ---
def load_tales_from_json(json_path: str) -> dict:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Успешно загружено {len(data['stories'])} сказок из JSON")
            return data
    except Exception as e:
        logger.error(f"Ошибка загрузки JSON: {e}")
        raise

def load_tests_from_json(json_path: str) -> dict:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Успешно загружено {len(data['tests'])} тестов из JSON")
            return data
    except Exception as e:
        logger.error(f"Ошибка загрузки тестов JSON: {e}")
        raise

def load_phonetics():
    try:
        with open("phonetics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки phonetics.json: {e}")
        return None

# Загружаем данные
try:
    tales_data = load_tales_from_json("fairytales.json")
except Exception as e:
    logger.critical(f"Не удалось загрузить данные: {e}")
    exit(1)

try:
    tests_data = load_tests_from_json("tests.json")
except Exception as e:
    logger.error(f"Не удалось загрузить тесты: {e}")
    tests_data = {"tests": []}

phonetics_data = load_phonetics()

# Загрузка данных
CULTURE_FILE = Path(__file__).parent / 'culture.json'

def load_culture_data():
    """Загрузка данных из culture.json с учётом структуры {'culture': [...]}"""
    try:
        if not CULTURE_FILE.exists():
            logger.error(f"Файл не найден: {CULTURE_FILE}")
            return []

        with open(CULTURE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Обрабатываем структуру {'culture': [...]}
            if isinstance(data, dict) and 'culture' in data:
                logger.info("Обнаружена структура с ключом 'culture'")
                return [
                    item for item in data['culture'] 
                    if isinstance(item, dict) 
                    and item.get('fact', '').strip()
                ]
            
            # Если структура не совпадает
            logger.error("Неверная структура файла culture.json")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Критическая ошибка загрузки: {e}")
        return []

# Явно инициализируем переменную
culture_data = load_culture_data()
logger.info(f"Загружено культурных фактов: {len(culture_data)}")












# --- Инициализация бота и диспетчера ---
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Инициализация базы данных
db = Database()



#Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)





async def set_bot_commands(bot: Bot):
    commands = [
        BotCommand(command="/start", description="Начать работу с ботом"),
        BotCommand(command="/menu", description="Главное меню"),
        BotCommand(command="/progress", description="Показать прогресс")
    ]
    await bot.set_my_commands(commands)

@dp.message(Command("menu"))
async def cmd_menu(message: types.Message):
    """Обработчик команды /menu с улучшенным персональным приветствием"""
    try:
        user = message.from_user
        name = ""
        # Формируем обращение в зависимости от доступных данных
        if user.first_name and user.last_name:
            name = f"{user.first_name} {user.last_name}"
        elif user.first_name:
            name = user.first_name
        elif user.last_name:
            name = user.last_name
        elif user.username:
            name = f"@{user.username}"
        else:
            name = "друг"
        # Регистрируем пользователя в базе данных
        db.add_user(user)
        # Форматируем текст с учетом возможного HTML-форматирования
        welcome_text = (
            f"🌟 <b>{html.escape(user.first_name)}</b>, ты в главном меню! \n \n"
            "Выбери <b>📖 Cказки</b>, если хочешь: \n"
            " • почитать или послушать сказки на хантыйском,\n"
            " • увидеть русский перевод сказки,\n"
            " • пройти тест на знание материала,\n\n"
             
            "Выбери <b>📚 Словарик</b>, если хочешь:\n"
            " • услышать произношение букв хантыйского алфавита,\n"
            " • увидеть список слов с переводом,\n"
            " • узнать грамматические правила. \n\n"
        )
        await message.answer(
            welcome_text,
            reply_markup=await main_menu_kb(),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"Ошибка в cmd_menu: {e}", exc_info=True)
        try:
            await message.answer(
                "🌟 Добро пожаловать в бота для изучения хантыйского языка!\n"
                "Пожалуйста, выбери раздел:",
                reply_markup=await main_menu_kb()
            )
        except Exception as fallback_error:
            logger.critical(f"Критическая ошибка в cmd_menu: {fallback_error}")























# --- Вспомогательные функции ---
async def split_long_message(text: str, max_length: int = 4096) -> List[str]:
    if len(text) <= max_length:
        return [text]
    parts = []
    while text:
        part = text[:max_length]
        split_pos = part.rfind('\n') if '\n' in part else max_length
        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    return parts

def build_menu(buttons: List[Tuple[str, str]], 
              back_button: Optional[Tuple[str, str]] = None,
              additional_buttons: List[Tuple[str, str]] = None,
              columns: int = 2) -> InlineKeyboardMarkup:
    """
    Создает inline клавиатуру с кнопками в несколько столбцов.
    
    :param buttons: Основные кнопки (текст, callback_data)
    :param back_button: Кнопка "Назад" (текст, callback_data)
    :param additional_buttons: Дополнительные кнопки (текст, callback_data)
    :param columns: Количество столбцов для основных кнопок (по умолчанию 2)
    :return: Объект InlineKeyboardMarkup
    """
    builder = InlineKeyboardBuilder()
    
    # Добавляем основные кнопки
    for text, data in buttons:
        builder.button(text=text, callback_data=data)
    
    # Добавляем дополнительные кнопки (например, описания)
    if additional_buttons:
        for text, data in additional_buttons:
            builder.button(text=text, callback_data=data)
    
    # Добавляем кнопку "Назад" если есть
    if back_button:
        builder.button(text=back_button[0], callback_data=back_button[1])
    
    # Настраиваем расположение кнопок:
    # 1. Основные кнопки - в указанное количество столбцов
    # 2. Дополнительные кнопки - по одной в ряд
    # 3. Кнопка "Назад" - отдельный ряд
    
    # Вычисляем количество строк для основных кнопок
    rows = (len(buttons) + columns - 1) // columns
    
    # Создаем список параметров для adjust:
    # - columns для каждой строки основных кнопок
    # - 1 для каждой дополнительной кнопки
    # - 1 для кнопки "Назад"
    adjust_params = [columns] * rows
    if additional_buttons:
        adjust_params.extend([1] * len(additional_buttons))
    if back_button:
        adjust_params.append(1)
    
    builder.adjust(*adjust_params)
    
    return builder.as_markup()






@dp.callback_query(F.data.startswith("show_culture_"))
async def show_culture_fact(callback: types.CallbackQuery, state: FSMContext):
    """Показывает культурный факт для сказки с возвратом к исходной версии"""
    try:
        story_id = int(callback.data.split("_")[-1])
        
        # Получаем текущее состояние (из какого языка пришли)
        user_data = await state.get_data()
        lang = user_data.get('last_lang', 'ru')  # По умолчанию русский
        
        culture_fact = next((cf for cf in culture_data if cf.get("id") == story_id and cf.get("fact")), None)
        
        if not culture_fact:
            await callback.answer("⚠️ Культурный факт не найден", show_alert=True)
            return
        
        # Формируем сообщение
        caption = f"🌿 <b>Культура</b>\n\n{culture_fact['fact']}"
        
        if culture_fact.get("source"):
            caption += f"\n\n🔗 Источник: {culture_fact['source']}"
        
        # Создаем кнопку возврата в зависимости от языка
        back_callback = f"{CALLBACK_LANGUAGE_RU}{story_id}" if lang == 'ru' else f"{CALLBACK_LANGUAGE_KH}{story_id}"
        
        kb = InlineKeyboardBuilder()
        kb.button(text="🔙 Назад к сказке", callback_data=back_callback)
        kb.button(text="🗂️ Главное меню", callback_data=CALLBACK_BACK_TO_MAIN)
        kb.adjust(2)
        
        if culture_fact.get("photo"):
            try:
                await callback.message.answer_photo(
                    photo=culture_fact["photo"],
                    caption=caption,
                    reply_markup=kb.as_markup()
                )
            except Exception as e:
                logger.error(f"Ошибка отправки фото: {e}")
                await callback.message.answer(
                    caption,
                    reply_markup=kb.as_markup()
                )
        else:
            await callback.message.answer(
                caption,
                reply_markup=kb.as_markup()
            )
            
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка показа культурного факта: {e}")
        await callback.answer("⚠️ Произошла ошибка при загрузке факта", show_alert=True)






# Функция для автоматического определения темы слова
@lru_cache(maxsize=5000)
def get_word_lemma(word: str) -> str:
    """Получить нормальную форму слова с кэшированием"""
    try:
        doc = Doc(word)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            return token.lemma.lower()
    except Exception:
        pass
    return word.lower().strip()

class ThemeClassifier:
    def __init__(self):
        self.theme_data = self._init_theme_data()
        self.compiled_patterns = self._compile_patterns()
        
    def _init_theme_data(self) -> Dict[str, Dict[str, Set[str]]]:
        return {
            # Животные
            'животные': {
                'exact': {
                    # Дикие животные
                    'медведь', 'лось', 'волк', 'лиса', 'заяц', 'росомаха', 'олень',
                    'выдра', 'белка', 'соболь', 'барсук', 'горностай', 'рысь', 'кот', 'дикий олень (букв.: лесной бык мужчина)',
                    # Птицы
                    'глухарь', 'тетерев', 'сова', 'ворон', 'дятел', 'сорока',
                    # Рыбы
                    'щука', 'налим', 'окунь', 'язь', 'плотва', 'карась',
                    # Насекомые
                    'комар', 'муха', 'пчела', 'бабочка', 'жук', 'мышонок (букв.: маленький мышонок сыночек)'
                },
                'patterns': [
                    r'животн', r'звер', r'птиц', r'рыб', 
                    r'насеком', r'млекопит'
                ]
            },
            
            # Природа
            'природа': {
                'exact': {
                    # Ландшафты
                    'тайга', 'тундра', 'степь', 'луг', 'поляна', 'равнина',
                    # Водоемы
                    'река', 'озеро', 'море', 'ручей', 'болото', 'родник',
                    # Горы
                    'гора', 'холм', 'сопка', 'утес', 'скала', 'пещера',
                    # Растительность
                    'дерево', 'береза', 'сосна', 'кедр', 'ель', 'пихта',
                    'кустарник', 'трава', 'мох', 'лишайник', 'папоротник',
                    # Ягоды и грибы
                    'брусника', 'черника', 'морошка', 'голубика', 'подберезовик',
                    # Погодные явления
                    'ветер', 'дождь', 'снег', 'град', 'туман', 'иней',
                    # Небесные тела
                    'солнце', 'луна', 'звезда', 'облако', 'радуга', 'закат', 'мороз'
                },
                'patterns': [
                    r'лес', r'вод', r'реч', r'озер', r'гор', 
                    r'растен', r'дерев', r'погод', r'неб', r'ясен'
                ]
            },
            
            # Люди
            'люди': {
                'exact': {
                    'мужчина', 'женщина', 'ребенок', 'старик', 'старуха',
                    'охотник', 'рыбак', 'мастер', 'шаман', 'знахарь',
                    'воин', 'вождь', 'путник', 'сосед', 'гость',
                    'учитель', 'ученик', 'родственник', 'незнакомец', 'хозяин', 'друг'
                },
                'patterns': [
                    r'человек', r'мужчин', r'женщин', r'ребен', 
                    r'стари', r'охот', r'рыбак', r'шаман'
                ]
            },
            
            # Семья и род
            'семья': {
                'exact': {
                    'семья', 'род', 'племя', 'родня', 'предок',
                    'отец', 'мать', 'сын', 'дочь', 'брат', 'сестра',
                    'дед', 'бабка', 'внук', 'внучка', 'дядя', 'тетя',
                    'свекор', 'тесть', 'зять', 'невестка', 'сноха'
                },
                'patterns': [
                    r'семь', r'род', r'плем', r'отц', 
                    r'матер', r'брат', r'сестр', r'предк'
                ]
            },
            
            # Части тела
            'части тела': {
                'exact': {
                    'голова', 'лицо', 'глаз', 'нос', 'рот', 'ухо',
                    'волосы', 'шея', 'плечо', 'рука', 'палец', 'нога',
                    'грудь', 'спина', 'живот', 'сердце', 'печень',
                    'кость', 'кровь', 'кожа', 'зуб', 'язык'
                },
                'patterns': [
                    r'голов', r'лиц', r'глаз', 
                    r'рот', r'ух', r'рук', r'ног'
                ]
            },
            
            # Числа и количество
            'числа': {
                'exact': {
                    # Основные числа
                    'один', 'два', 'три', 'четыре', 'пять',
                    'шесть', 'семь', 'восемь', 'девять', 'десять',
                    # Десятки
                    'двадцать', 'тридцать', 'сорок', 'пятьдесят',
                    # Большие числа
                    'сто', 'двести', 'пятьсот', 'тысяча',
                    # Дробные
                    'половина', 'треть', 'четверть',
                    # Количественные
                    'много', 'мало', 'несколько', 'пара', 'десяток'
                },
                'patterns': [
                    r'числ', r'колич', r'один', r'два', 
                    r'три', r'четыр', r'пят', r'десят'
                ]
            },
            
            # Время
            'время': {
                'exact': {
                    # Времена года
                    'зима', 'весна', 'лето', 'осень',
                    # Месяцы
                    'январь', 'февраль', 'март', 'апрель',
                    # Части суток
                    'утро', 'день', 'вечер', 'ночь',
                    # Понятия
                    'год', 'месяц', 'неделя', 'час', 'минута',
                    'вчера', 'сегодня', 'завтра', 'сейчас'
                },
                'patterns': [
                    r'врем', r'год', r'месяц', r'недел',
                    r'час', r'утр', r'день', r'вечер'
                ]
            },
            
            # Действия
            'действия': {
                'exact': {
                    # Базовые
                    'идти', 'бежать', 'стоять', 'сидеть', 'лежать',
                    # Работа
                    'делать', 'работать', 'строить', 'копать', 'рубить',
                    # Взаимодействие
                    'говорить', 'слушать', 'видеть', 'смотреть', 'думать', 'если не знал (букв. не если знал)',
                    # Социальные
                    'давать', 'брать', 'помогать', 'бить', 'целовать', 'шептать', 'хвастаться', 'хвастать, хвалиться', 'жили (вдвоем)',
                    # Охота
                    'охотиться', 'ловить', 'стрелять', 'собирать', 'резать', 'шептать, говорить себе по нос', 'хвастать', 'выйди', 'жили', 'танцевать'
                },
                'patterns': [
                    r'дел', r'работ', r'говор', r'слуш',
                    r'вид', r'смотр', r'дум', r'ход'
                ]
            },

            'местоимения': {
                'exact': {
                    # Личные местоимения
                    'я', 'ты', 'он', 'она', 'оно',
                    'мы', 'вы', 'они', 'ему', 'ей',
                    
                    # Возвратное
                    'себя',
                    
                    # Притяжательные
                    'мой', 'твой', 'его', 'её', 'наш',
                    'ваш', 'их', 'свой',
                    
                    # Указательные
                    'этот', 'тот', 'такой', 'столько',
                    'этакий', 'таков', 'сей', 'оный',
                    
                    # Определительные
                    'весь', 'всякий', 'каждый', 'любой',
                    'сам', 'самый', 'иной', 'другой',
                    'целый', 'цельный',
                    
                    # Вопросительные
                    'кто', 'что', 'какой', 'который',
                    'чей', 'сколько',
                    
                    # Относительные (те же, что вопросительные)
                    'кто', 'что', 'какой', 'который',
                    'чей', 'сколько',
                    
                    # Отрицательные
                    'никто', 'ничто', 'никакой',
                    'ничей', 'некого', 'нечего',
                    
                    # Неопределенные
                    'некто', 'нечто', 'некоторый',
                    'некий', 'кое-кто', 'кое-что',
                    'кто-то', 'что-то', 'какой-то',
                    'чей-то', 'сколько-то',
                    'кто-нибудь', 'что-нибудь',
                    'какой-нибудь', 'чей-нибудь',
                    'сколько-нибудь',
                    'кто-либо', 'что-либо',
                    'какой-либо', 'чей-либо',
                    'сколько-либо'
                },
                'patterns': [
                    r'\bя\b', r'\bты\b', r'\bон\b', r'\bона\b', r'\bоно\b',
                    r'\bмы\b', r'\bвы\b', r'\bони\b',
                    r'\bсеб\w*',  # себя, себе, собою
                    r'\bм[оё]й\b', r'\bтв[оё]й\b', r'\bсв[оё]й\b',
                    r'\bнаш\b', r'\bваш\b', r'\bих\b',
                    r'\bэт\w*', r'\bт\w*',  # этот, тот, такая
                    r'\bкто\b', r'\bчто\b', r'\bкак\w*', r'\bкото\w*',
                    r'\bчей\b', r'\bскольк\w*',
                    r'\bникт\w*', r'\bничт\w*', r'\bникак\w*',
                    r'\bнект\w*', r'\bнечт\w*', r'\bнекот\w*',
                    r'\bкое-\w*', r'\b\w+-то\b', r'\b\w+-нибудь\b', r'\b\w+-либо\b'
                ]
            },
                        
            # Жилище и быт
            'жилище': {
                'exact': {
                    'дом', 'жилище', 'чум', 'шалаш', 'землянка',
                    'печь', 'костер', 'дверь', 'окно', 'порог',
                    'посуда', 'котел', 'ковш', 'нож', 'топор',
                    'одежда', 'обувь', 'шапка', 'пояс', 'игла'
                },
                'patterns': [
                    r'жил', r'дом', r'постр', r'печ',
                    r'посу', r'одеж', r'обув', r'инструмент'
                ]
            },

            'качества и состояния': {
                'exact': {
                    # Физические характеристики
                    'тяжелый', 'легкий', 'большой', 'маленький', 'крепкий', 'хрупкий',
                    'горячий', 'холодный', 'влажный', 'сухой', 'сильный', 'милый', 'тяжело',
                    
                    # Эмоциональные состояния
                    'радостный', 'грустный', 'страшный', 'смешной',
                    
                    # Оценочные характеристики
                    'хороший', 'плохой', 'красивый', 'уродливый',
                    
                    # Сложность
                    'трудно', 'легко', 'сложно', 'просто',
                    
                    # Скорость
                    'быстро', 'медленно', 'резко', 'плавно'
                },
                'patterns': [
                    r'тяжел', r'лёгк', r'больш', r'маленьк',
                    r'горяч', r'холодн', r'радост', r'грустн',
                    r'хорош', r'плох', r'красив', r'уродлив',
                    r'трудн', r'легк', r'сложн', r'прост',
                    r'быстр', r'медлен', r'резк', r'плавн'
                ]
            },
            'базовые слова': {
                'exact': {
                    # Утверждения и отрицания
                    'да', 'нет', 'не', 'ни', 'никак', 'нисколько',
                    
                    # Вопросительные слова
                    'кто', 'что', 'какой', 'чей', 'где', 'куда',
                    'откуда', 'когда', 'зачем', 'почему', 'как',
                    'сколько', 'насколько', 'отчего',
                    
                    # Указательные слова
                    'вот', 'вон', 'тут', 'там', 'здесь', 'туда',
                    'сюда', 'оттуда', 'отсюда',
                    
                    # Модальные частицы
                    'ли', 'разве', 'неужели', 'ведь', 'же',
                    'бы', 'пусть', 'давай', 'давайте',
                    
                    # Союзы
                    'и', 'а', 'но', 'или', 'чтобы', 'потому что',
                    'если', 'хотя', 'так как'
                },
                'patterns': [
                    # Утверждения/отрицания
                    r'\bда\b', r'\bнет\b', r'\bне\b', r'\bни\b',
                    
                    # Вопросы
                    r'\bкто\b', r'\bчто\b', r'\bкак\w*', r'\bгде\b',
                    r'\bкуда\b', r'\bкогда\b', r'\bпочему\b', r'\bзачем\b',
                    
                    # Указатели
                    r'\bвот\b', r'\bвон\b', r'\bтут\b', r'\bтам\b',
                    
                    # Частицы
                    r'\bли\b', r'\bразве\b', r'\bнеужели\b', r'\bведь\b',
                    
                    # Союзы
                    r'\bи\b', r'\bа\b', r'\bно\b', r'\bили\b'
                ]
            },
            # Духовная культура
            'духовная культура': {
                'exact': {
                    'дух', 'бог', 'тотем', 'оберег', 'амулет',
                    'шаман', 'колдун', 'предсказатель', 'праздник',
                    'обряд', 'ритуал', 'песня', 'сказка', 'легенда',
                    'запрет', 'табу', 'обычай', 'традиция'
                },
                'patterns': [
                    r'дух', r'бог', r'шаман', r'обряд',
                    r'ритуал', r'праздн', r'легенд', r'традиц'
                ]
            }
        }

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Компиляция regex-паттернов для производительности"""
        compiled = {}
        for theme, data in self.theme_data.items():
            patterns = [re.compile(p) for p in data['patterns']]
            compiled[theme] = patterns
        return compiled
    
    def detect_theme(self, word: str) -> str:
        """Определение темы слова с приоритетами"""
        if not word or not isinstance(word, str):
            return "Общее"
            
        word_clean = word.lower().strip()
        if not word_clean:
            return "Общее"
        
        word_lemma = get_word_lemma(word_clean)
        
        # 1. Проверка точных совпадений
        for theme, data in self.theme_data.items():
            if word_clean in data['exact'] or word_lemma in data['exact']:
                return theme.capitalize()
        
        # 2. Проверка по паттернам
        for theme, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(word_clean) or pattern.search(word_lemma):
                    return theme.capitalize()
        
        # 3. Морфологический анализ для неизвестных слов
        doc = Doc(word_clean)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        
        for token in doc.tokens:
            if 'NOUN' in token.pos:
                if 'Animacy=Anim' in token.feats:
                    return "Животные"
                return "Природа"
            elif 'VERB' in token.pos:
                return "Действия"
            elif 'ADJ' in token.pos:
                return "Характеристики"
        
        return "Общее"






async def send_audio_if_exists(chat_id: int, story: dict):
    """Отправляет аудиофайл, если он существует"""
    if story.get('audio') and story['audio'] != "pass":
        audio_path = Path(__file__).parent / "audio" / story['audio']
        try:
            if audio_path.exists():
                # Правильное создание InputFile
                audio_file = types.FSInputFile(audio_path)
                await bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    title=f"{story['rus_title']} | {story['han_title']}",
                    performer="Хантыйская сказка",
                    caption=f"🎧 {story['rus_title']}"
                )
                return True
        except Exception as e:
            logger.error(f"Ошибка при отправке аудио: {e}")
    return False


async def send_question(message: types.Message, question: dict, current: int, total: int):
    """Отправляет вопрос теста"""
    builder = InlineKeyboardBuilder()
    for i, variant in enumerate(question["variants"]):
        builder.button(text=variant, callback_data=f"test_answer_{question['q_id']}_{i}")
    builder.adjust(1)
    await message.answer(
        f"📝 Вопрос {current + 1}/{total}\n"
        f"{question['question']}",
        reply_markup=builder.as_markup()
    )


async def alphabet_menu_kb() -> InlineKeyboardMarkup:
    """Меню раздела алфавита"""
    buttons = [
        ("🔠 Названия букв", CALLBACK_ALPHABET_LETTERS),
        ("🔡 Гласные звуки", CALLBACK_ALPHABET_VOWELS),
        ("🔣 Согласные звуки", CALLBACK_ALPHABET_CONSONANTS)
    ]
    return build_menu(buttons, ("🔙 Назад", CALLBACK_BACK_TO_VOCABULARY), columns=1)


# --- Клавиатуры ---
async def main_menu_kb() -> InlineKeyboardMarkup:
    """Главное меню"""
    buttons = [
        ("📖 Сказки", CALLBACK_TALES),
        ("📚 Словарик", CALLBACK_VOCABULARY),
        ("📊 Мой прогресс", CALLBACK_PROGRESS)
    ]
    return build_menu(buttons, columns=2)


async def vocabulary_menu_kb() -> InlineKeyboardMarkup:
    """Меню словаря"""
    buttons = [
        ("📝 Общая грамматика", CALLBACK_GRAMMAR),
        ("🔤 Общая лексика", CALLBACK_LEXICON),
        ("🔡 Алфавит", CALLBACK_ALPHABET)
    ]
    return build_menu(buttons, ("🗂️ Главное меню", CALLBACK_BACK_TO_MAIN), columns=2)






















async def tales_menu_kb(page: int = 0, page_size: int = 5) -> InlineKeyboardMarkup:
    """Меню сказок с пагинацией"""
    stories = tales_data['stories']
    total_pages = (len(stories) + page_size - 1) // page_size
    start_idx = page * page_size
    end_idx = start_idx + page_size
    paginated_stories = stories[start_idx:end_idx]
    buttons = [
        (story['rus_title'], f"{CALLBACK_SHOW_STORY}{story['id']}") 
        for story in paginated_stories
    ]
    navigation_buttons = []
    if page > 0:
        navigation_buttons.append(("◀️ Назад", f"{CALLBACK_TALES_PAGE_PREFIX}{page-1}"))
    if end_idx < len(stories):
        navigation_buttons.append(("Вперед ▶️", f"{CALLBACK_TALES_PAGE_PREFIX}{page+1}"))
    return build_menu(
        buttons, 
        back_button=("🗂️ Главное меню", CALLBACK_BACK_TO_MAIN),
        additional_buttons=navigation_buttons,
        columns=1
    )


async def language_menu_kb(story_id: int) -> InlineKeyboardMarkup:
    """Меню выбора языка для сказки"""
    buttons = [
        ("🇷🇺 Русский", f"{CALLBACK_LANGUAGE_RU}{story_id}"),
        ("🦦 Хантыйский", f"{CALLBACK_LANGUAGE_KH}{story_id}")
    ]
    return build_menu(buttons, ("🔙 Назад", CALLBACK_BACK_TO_TALES), columns=2)










async def story_menu_kb(story_id: int) -> InlineKeyboardMarkup:
    """Меню для конкретной сказки - кнопки только если есть данные"""
    try:
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        buttons = []
        has_illustrations = os.path.exists(f"illustraciones/{story['rus_title']}") and any(os.scandir(f"illustraciones/{story['rus_title']}"))
        has_audio = story.get('audio') and os.path.exists(f"audio/{story['audio']}")
        has_grammar = bool(story.get('grammar', '').strip())
        has_lexicon = bool(story.get('han_words')) and bool(story.get('rus_words'))
        # Детальная проверка культурного факта с логированием
        has_culture = False
        for cf in culture_data:
            try:
                cf_id = int(cf.get('id', -1))
                cf_fact = cf.get('fact', '').strip()
                if cf_id == story_id and cf_fact:
                    has_culture = True
                    logger.info(f"Найден культурный факт для story_id={story_id}: {cf_fact[:50]}...")
                    break
            except Exception as e:
                logger.warning(f"Ошибка обработки культурного факта: {e}")
        
        logger.info(f"Итог проверки для story_id={story_id}: has_culture={has_culture}")
        
     
         # Формирование кнопок
        if has_illustrations:
            buttons.append(("🖼️ Иллюстрации", f"{CALLBACK_SHOW_ILLUSTRATIONS}{story_id}"))
        if has_audio:
            buttons.append(("🎧 Аудио", f"{CALLBACK_PLAY_AUDIO}{story_id}"))
        if has_grammar:
            buttons.append(("📖 Грамматика", f"{CALLBACK_SHOW_GRAMMAR}{story_id}"))
        if has_lexicon:
            buttons.append(("🔤 Лексика", f"{CALLBACK_SHOW_LEXICON}{story_id}"))
        if any(t["fairytale_id"] == story_id for t in tests_data["tests"]):
            buttons.append(("📝 Пройти тест", f"start_test_{story_id}"))
        if has_culture:
            buttons.append(("🌿 Культура", f"show_culture_{story_id}"))
        
        
        return build_menu(buttons, ("🔙 Назад", CALLBACK_BACK_TO_TALES), columns=2)
    
    except Exception as e:
        logger.error(f"Ошибка в story_menu_kb: {e}")
        return build_menu([], ("🔙 Назад", CALLBACK_BACK_TO_TALES))




















# --- Обработчики сообщений ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start с улучшенным персональным приветствием"""
    try:
        user = message.from_user
        name = ""
        # Формируем обращение в зависимости от доступных данных
        if user.first_name and user.last_name:
            name = f"{user.first_name} {user.last_name}"
        elif user.first_name:
            name = user.first_name
        elif user.last_name:
            name = user.last_name
        elif user.username:
            name = f"@{user.username}"
        else:
            name = "друг"
        # Регистрируем пользователя в базе данных
        db.add_user(user)
        # Форматируем текст с учетом возможного HTML-форматирования
        welcome_text = (
            f"🌟 Вўща, <b>{html.escape(name)}</b> 🐾\n \n"
            "Добро пожаловать в чат-бот для изучения казымского диалекта хантыйского языка!\n\n"
            "<b>Здесь ты сможешь:</b>\n"
            "   • 📖 Прочитать сказки на хантыйском и русском\n"
            "   • 📚 Изучить слова и грамматику\n"
            "   • 🔤 Познакомиться с алфавитом и фонетикой\n\n"
            "<b>Выбери интересующий раздел:</b>"
        )
        await message.answer(
            welcome_text,
            reply_markup=await main_menu_kb(),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"Ошибка в cmd_start: {e}", exc_info=True)
        try:
            await message.answer(
                "🌟 Добро пожаловать в бота для изучения хантыйского языка!\n"
                "Пожалуйста, выбери раздел:",
                reply_markup=await main_menu_kb()
            )
        except Exception as fallback_error:
            logger.critical(f"Критическая ошибка в cmd_start: {fallback_error}")


@dp.callback_query(F.data == CALLBACK_PROGRESS)
@dp.message(Command("progress"))
async def show_progress(update: Union[types.Message, types.CallbackQuery]):
    """Универсальный обработчик для команды /progress и кнопки прогресса"""
    try:
        # Получаем сообщение и пользователя
        if isinstance(update, types.CallbackQuery):
            message = update.message
            user = update.from_user
            is_callback = True
        else:
            message = update
            user = update.from_user
            is_callback = False

        # Получаем прогресс из базы данных
        with closing(sqlite3.connect(db.db_name)) as conn:
            cursor = conn.cursor()
            
            # Получаем базовую статистику
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT tale_id) as tales_read,
                    SUM(read_count) as total_reads,
                    COUNT(DISTINCT CASE WHEN completed THEN tale_id END) as tales_completed
                FROM tale_progress
                WHERE user_id = ?
            """, (user.id,))
            stats = cursor.fetchone()
            
            tales_read = stats[0] if stats and stats[0] is not None else 0
            total_reads = stats[1] if stats and stats[1] is not None else 0
            tales_completed = stats[2] if stats and stats[2] is not None else 0

            # Получаем все завершенные сказки
            cursor.execute("""
                SELECT tale_id, read_count
                FROM tale_progress
                WHERE user_id = ? AND completed = TRUE
                ORDER BY last_read_date DESC
            """, (user.id,))
            completed_tales = cursor.fetchall()

            # Получаем последние 5 прочитанных сказок
            cursor.execute("""
                SELECT tale_id, last_read_date, read_count, completed
                FROM tale_progress
                WHERE user_id = ?
                ORDER BY last_read_date DESC
                LIMIT 5
            """, (user.id,))
            recent_tales = cursor.fetchall()

        # Формируем текст с прогрессом
        progress_text = (
            f"📊 <b>Ваш прогресс:</b>\n"
            f"      •📜 Прочитано сказок: <b>{tales_read}</b>\n"
            f"      •🔁 Всего прочтений: <b>{total_reads}</b>\n"
            f"      •🏁 Завершено тестов: <b>{tales_completed}</b>\n\n"
        )
        
        # Добавляем список завершенных сказок
        if completed_tales:
            progress_text += "<b>✅ Завершённые сказки:</b>\n"
            for tale in completed_tales:
                tale_id, read_count = tale
                story = next((s for s in tales_data['stories'] if s['id'] == tale_id), None)
                if story:
                    progress_text += f"     •🗞️ <b>{story['rus_title']}</b> (прочитано {read_count} раз(а))\n"
            progress_text += "\n\n"
        
        # Добавляем информацию о недавно прочитанных сказках
        if recent_tales:
            progress_text += "<b>📚 Недавно прочитанные:</b>\n"
            for tale in recent_tales:
                tale_id, last_read, read_count, completed = tale
                story = next((s for s in tales_data['stories'] if s['id'] == tale_id), None)
                if story:
                    status = "📗" if completed else "📖"
                    progress_text += (
                        f"      •{status} <b>{story['rus_title']}</b> - "
                        f"прочитано {read_count} раз(а)\n"
                    )
        else:
            progress_text += "Вы еще не читали сказки\n"

        # Создаем клавиатуру
        builder = InlineKeyboardBuilder()
        builder.button(text="🗂️ Главное меню", callback_data=CALLBACK_BACK_TO_MAIN)
        
        # Отправляем сообщение
        if is_callback:
            try:
                await message.answer(
                    progress_text,
                    reply_markup=builder.as_markup(),
                    parse_mode=ParseMode.HTML
                )
            except:
                await message.answer(
                    progress_text,
                    reply_markup=builder.as_markup(),
                    parse_mode=ParseMode.HTML
                )
            await update.answer()
        else:
            await message.answer(
                progress_text,
                reply_markup=builder.as_markup(),
                parse_mode=ParseMode.HTML
            )
            
    except Exception as e:
        logger.error(f"Ошибка в show_progress: {e}", exc_info=True)
        error_msg = "⚠️ Ошибка при загрузке прогресса"
        if isinstance(update, types.CallbackQuery):
            await update.answer(error_msg, show_alert=True)
        else:
            await message.answer(error_msg)















# --- Обработчики сказок ---
@dp.callback_query(F.data == CALLBACK_TALES)
async def handle_tales_first(callback: types.CallbackQuery):
    """Первый вход в меню сказок — создает новое сообщение"""
    try:
        page = 0
        await callback.message.answer(
            "📖 Выбери сказку на этой страничке или нажми <b>Вперёд ▶️</b>, чтобы увидеть другие:",
            reply_markup=await tales_menu_kb(page=page)
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_tales_first: {e}")
        await callback.answer("⚠️ Ошибка при загрузке меню", show_alert=True)


@dp.callback_query(F.data.startswith(CALLBACK_TALES_PAGE_PREFIX))
async def handle_tales_pagination(callback: types.CallbackQuery):
    """Обработчик пагинации в меню сказок — редактирует текущее сообщение"""
    try:
        page = int(callback.data.replace(CALLBACK_TALES_PAGE_PREFIX, ""))
        await callback.message.edit_text(
            "📖 Выбери сказку или воспользуйся кнопками <b>Вперёд ▶️</b> и <b>◀️ Назад</b> для перехода по меню:",
            reply_markup=await tales_menu_kb(page=page)
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_tales_pagination: {e}")
        await callback.answer("⚠️ Ошибка при загрузке меню", show_alert=True)


@dp.callback_query(F.data.startswith(CALLBACK_SHOW_STORY))
async def handle_show_story(callback: types.CallbackQuery):
    """Выбор языка для сказки"""
    try:
        story_id = int(callback.data.replace(CALLBACK_SHOW_STORY, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        await callback.message.answer(
            f"📖 <b>{story['rus_title']}</b>\nВыбери язык:",
            reply_markup=await language_menu_kb(story_id)
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_show_story: {e}")
        await callback.answer("⚠️ Ошибка при загрузке сказки", show_alert=True)


@dp.callback_query(F.data.startswith(CALLBACK_LANGUAGE_RU))
async def handle_language_ru(callback: types.CallbackQuery, state: FSMContext):
    story_id = int(callback.data.replace(CALLBACK_LANGUAGE_RU, ""))
    await state.update_data(last_lang='ru')
    """Показ сказки на русском (только русское название)"""
    try:
        story_id = int(callback.data.replace(CALLBACK_LANGUAGE_RU, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        
        # Обновляем прогресс пользователя и получаем статус обновления
        was_updated = db.update_tale_progress(callback.from_user.id, story_id)
        
        # Только русское название
        message = f"📖 <b>{story['rus_title']}</b>\n{story['rus_text']}"
        parts = await split_long_message(message)
        
        # Добавляем сообщение о прогрессе, если это не первое прочтение
        if was_updated:
            progress = db.get_user_progress(callback.from_user.id)
            for tale in progress["recent_tales"]:
                if tale[0] == story_id:
                    read_count = tale[2]
                    message = f"📖 <b>{story['rus_title']}</b> (прочитано {read_count} раз(а))\n{story['rus_text']}"
                    parts = await split_long_message(message)
                    break
        
        # Отправляем первую часть без кнопок
        for part in parts[:-1]:
            await callback.message.answer(part)
        
        # Отправляем остальные части
        await callback.message.answer(
            parts[-1],
            reply_markup=await story_menu_kb(story_id))

                
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_language_ru: {e}")
        await callback.answer("⚠️ Ошибка при загрузке сказки", show_alert=True)

@dp.callback_query(F.data.startswith(CALLBACK_LANGUAGE_KH))
async def handle_language_kh(callback: types.CallbackQuery, state: FSMContext):
    story_id = int(callback.data.replace(CALLBACK_LANGUAGE_KH, ""))
    await state.update_data(last_lang='kh')
    """Показ сказки на хантыйском (с хантыйским и русским названием)"""
    try:
        story_id = int(callback.data.replace(CALLBACK_LANGUAGE_KH, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        
        # Обновляем прогресс пользователя и получаем статус обновления
        was_updated = db.update_tale_progress(callback.from_user.id, story_id)
        
        # Хантыйское + русское название
        message = (
            f"📖 <b>{story['han_title']}</b>\n"
            f"<i>({story['rus_title']})</i>\n"
            f"{story['han_text']}"
        )
        parts = await split_long_message(message)
        
        # Добавляем сообщение о прогрессе, если это не первое прочтение
        if was_updated:
            progress = db.get_user_progress(callback.from_user.id)
            for tale in progress["recent_tales"]:
                if tale[0] == story_id:
                    read_count = tale[2]
                    message = (
                        f"📖 <b>{story['han_title']}</b> (прочитано {read_count} раз(а))\n"
                        f"<i>({story['rus_title']})</i>\n"
                        f"{story['han_text']}"
                    )
                    parts = await split_long_message(message)
                    break
        
        # Отправляем первую часть без кнопок
        for part in parts[:-1]:
            await callback.message.answer(part)
        
        # Отправляем остальные части
        await callback.message.answer(
            parts[-1],
            reply_markup=await story_menu_kb(story_id))
                
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_language_kh: {e}")
        await callback.answer("⚠️ Ошибка при загрузке сказки", show_alert=True)

@dp.callback_query(F.data.startswith(CALLBACK_PLAY_AUDIO))
async def handle_play_audio(callback: types.CallbackQuery):
    """Обработчик кнопки аудио - отправляет ТОЛЬКО аудио"""
    try:
        story_id = int(callback.data.replace(CALLBACK_PLAY_AUDIO, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        if story.get('audio') and story['audio'] != "pass":
            audio_path = Path(__file__).parent / "audio" / story['audio']
            if audio_path.exists():
                await bot.send_audio(
                    chat_id=callback.message.chat.id,
                    audio=types.FSInputFile(audio_path),
                    title=f"{story['rus_title']} | {story['han_title']}",
                    performer="Хантыйская сказка",
                    caption=f"🎧 {story['rus_title']}"
                )
            else:
                await callback.answer("⚠️ Аудиофайл не найден", show_alert=True)
        else:
            await callback.answer("⚠️ Для этой сказки нет аудио", show_alert=True)
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_play_audio: {e}")
        await callback.answer("⚠️ Ошибка при загрузке аудио", show_alert=True)


@dp.callback_query(F.data.startswith(CALLBACK_SHOW_GRAMMAR))
async def handle_show_grammar(callback: types.CallbackQuery):
    """Показ грамматики для конкретной сказки (с проверкой)"""
    try:
        story_id = int(callback.data.replace(CALLBACK_SHOW_GRAMMAR, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)

        # Проверяем наличие грамматики
        if not story.get('grammar') or not story['grammar'].strip():
            await callback.answer("❌ Для этой сказки нет грамматики", show_alert=True)
            return

        message = f"📝 <b>Грамматика для сказки '{story['rus_title']}':</b>\n{story['grammar']}"
        parts = await split_long_message(message)
        await callback.message.answer(
            parts[0],
            reply_markup=await story_menu_kb(story_id)
        )
        for part in parts[1:]:
            await callback.message.answer(part)
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_show_grammar: {e}")
        await callback.answer("⚠️ Ошибка при загрузке грамматики", show_alert=True)


@dp.callback_query(F.data.startswith(CALLBACK_SHOW_LEXICON))
async def handle_show_lexicon(callback: types.CallbackQuery):
    """Показ лексики для конкретной сказки (с проверкой)"""
    try:
        story_id = int(callback.data.replace(CALLBACK_SHOW_LEXICON, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)

        # Проверяем наличие лексики
        if (not story.get('han_words') or not story.get('rus_words') or
            len(story['han_words']) == 0 or len(story['rus_words']) == 0):
            await callback.answer("❌ Для этой сказки нет лексики", show_alert=True)
            return

        # Проверяем совпадение количества слов
        if len(story['han_words']) != len(story['rus_words']):
            logger.warning(f"Несоответствие количества слов в сказке {story_id}")

        word_pairs = []
        for han, rus in zip(story['han_words'], story['rus_words']):
            word_pairs.append(f"• <b>{han}</b> - {rus}")

        message = f"🔤 <b>Лексика для сказки '{story['rus_title']}':</b>\n" + "\n".join(word_pairs)
        parts = await split_long_message(message)
        await callback.message.answer(
            parts[0],
            reply_markup=await story_menu_kb(story_id)
        )
        for part in parts[1:]:
            await callback.message.answer(part)
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_show_lexicon: {e}")
        await callback.answer("⚠️ Ошибка при загрузке лексики", show_alert=True)


@dp.callback_query(F.data.startswith(CALLBACK_BACK_TO_LANGUAGE))
async def handle_back_to_language(callback: types.CallbackQuery):
    """Возврат к выбору языка"""
    try:
        story_id = int(callback.data.replace(CALLBACK_BACK_TO_LANGUAGE, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        await callback.message.answer(
            f"📖 <b>{story['rus_title']}</b>\nВыберите язык:",
            reply_markup=await language_menu_kb(story_id)
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_back_to_language: {e}")
        await callback.answer("⚠️ Ошибка при возврате к выбору языка", show_alert=True)







# --- Обработчики тестов ---
@dp.callback_query(F.data.startswith("start_test_"))
async def handle_start_test(callback: types.CallbackQuery, state: FSMContext):
    """Начало теста по сказке"""
    try:
        tale_id = int(callback.data.replace("start_test_", ""))
        test = next((t for t in tests_data["tests"] if t["fairytale_id"] == tale_id), None)
        if not test or not test["questions"]:
            await callback.answer("Для этой сказки пока нет теста", show_alert=True)
            return

        # Сохраняем текущий тест в состоянии пользователя
        await state.set_data({
            "current_test": test,
            "current_question": 0,
            "test_score": 0
        })

        # Отправляем первый вопрос
        await send_question(callback.message, test["questions"][0], 0, len(test["questions"]))
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_start_test: {e}", exc_info=True)
        await callback.answer("⚠️ Ошибка при запуске теста", show_alert=True)









@dp.callback_query(F.data.startswith("test_answer_"))
async def handle_test_answer(callback: types.CallbackQuery, state: FSMContext):
    """Обработка ответа на вопрос теста"""
    try:
        parts = callback.data.split("_")
        q_id = int(parts[2])
        answer_idx = int(parts[3])
        
        # Получаем данные из FSMContext
        user_data = await state.get_data()
        test = user_data.get("current_test")
        current_question = user_data.get("current_question", 0)
        test_score = user_data.get("test_score", 0)
        answered_with_mistake = user_data.get("answered_with_mistake", set())

        if not test:
            await callback.answer("Тест не найден", show_alert=True)
            return

        question = test["questions"][current_question]
        selected_answer = question["variants"][answer_idx]
        right_answer = question["right answer"]

        # Поддержка нескольких правильных ответов (список или строка)
        if isinstance(right_answer, list):
            right_answers = [str(ans).strip().lower() for ans in right_answer]
            is_correct = str(selected_answer).strip().lower() in right_answers
        else:
            is_correct = str(selected_answer).strip().lower() == str(right_answer).strip().lower()

        # Сохраняем результат в базу данных
        db.save_test_result(
            user_id=callback.from_user.id,
            tale_id=test["fairytale_id"],
            question_id=q_id,
            is_correct=is_correct
        )

        explanation = question.get('explanation', 'Объяснение отсутствует.')

        # Если ответ неверный
        if not is_correct:
            # Запоминаем, что была ошибка
            answered_with_mistake.add(current_question)
            await state.update_data(answered_with_mistake=answered_with_mistake)
            
            # Показываем алёрт с ошибкой
            await callback.answer(f"❌ Неверно.\nПопробуйте снова.", show_alert=True)
            return

        # Если ответ верный
        if current_question not in answered_with_mistake:
            # Ответ верный с первого раза - засчитываем полный балл
            test_score += 1
            # Показываем сообщение с пояснением
            await callback.message.answer(f"✅ Верно!\n{explanation}")
        else:
            # Ответ верный, но после ошибки - засчитываем 0.5 балла
            test_score += 0.5
            # Показываем сообщение с пояснением
            await callback.message.answer(f"✅ Теперь верно.\n{explanation}")

        # Обновляем данные пользователя
        await state.update_data({
            "current_test": test,
            "current_question": current_question + 1,
            "test_score": test_score,
            "answered_with_mistake": set()  # Сбрасываем для нового вопроса
        })

        # Переход к следующему вопросу или завершение
        if current_question + 1 < len(test["questions"]):
            await send_question(
                callback.message,
                test["questions"][current_question + 1],
                current_question + 1,
                len(test["questions"])
            )
        else:
            score_percent = int((test_score / len(test["questions"])) * 100)
            tale = next(t for t in tales_data["stories"] if t["id"] == test["fairytale_id"])
            
            # Добавляем вызов mark_tale_completed если тест пройден успешно
            if score_percent >= 70:  # Порог успешного прохождения теста
                db.mark_tale_completed(callback.from_user.id, test["fairytale_id"])
            
            completion_msg = "🎉 Поздравляем! Вы успешно прошли тест." if score_percent >= 70 else "Вы можете пройти тест ещё раз."
            await callback.message.answer(
                f"📊 Тест по сказке '{tale['rus_title']}' завершён!\n"
                f"Ваш результат: {test_score:.1f} из {len(test['questions'])} ({score_percent}%)\n"
                f"{completion_msg}",
                reply_markup=await story_menu_kb(test["fairytale_id"])
            )

        await callback.answer()

    except Exception as e:
        logger.error(f"Ошибка в handle_test_answer: {e}", exc_info=True)
        await callback.answer("⚠️ Ошибка при обработке ответа", show_alert=True)


        






# --- Обработчики словаря ---
@dp.callback_query(F.data == CALLBACK_VOCABULARY)
async def handle_vocabulary(callback: types.CallbackQuery):
    """Обработчик раздела словаря"""
    try:
        await callback.message.answer(
            "📚 Выбери раздел словаря:\n\n"
            "В <b>📝 Общей грамматике</b> можешь прочитать о грамматических правилах: \n" 
            " • Сколько чисел в хантыйском и как они образуются,\n "
            " • Какие есть падежные суффиксы,\n"
            " • Как ласково сказать белочка или рыбка.\n\n"

            "В <b>🔤 Общей лексике</b> сможешь узнать слова из разных категорий:\n" 
            " • Еда,\n" 
            " • Животные,\n"
            " • Природа и другие.\n\n"

            "В <b>🔡 Алфавите</b> можешь увидеть:\n" 
            " • Названия букв\n" 
            " • Гласные звуки\n" 
            " • Согласные звуки.\n",
            reply_markup=await vocabulary_menu_kb()
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_vocabulary: {e}")
        await callback.answer("⚠️ Ошибка при загрузке меню", show_alert=True)


@dp.callback_query(F.data == CALLBACK_GRAMMAR)
async def handle_grammar(callback: types.CallbackQuery):
    """Показ общей грамматики"""
    try:
        grammar_parts = []
        for story in tales_data['stories']:
            if story.get('grammar'):
                grammar_parts.append(f"📝 <b>{story['rus_title']}</b>\n{story['grammar']}\n")

        if not grammar_parts:
            await callback.message.answer("❌ Информация по грамматике не найдена")
            return

        full_message = "\n".join(grammar_parts)
        parts = await split_long_message(full_message)

        # Редактируем текущее сообщение (вместо удаления)
        if len(parts) > 0:
            await callback.message.answer(
                parts[0],
                reply_markup=build_menu([], ("🔙 Назад", CALLBACK_BACK_TO_VOCABULARY))
            )

        # Отправляем остальные части как новые сообщения
        for part in parts[1:]:
            await callback.message.answer(part)

        await callback.answer()
    except AiogramError as e:
        logger.error(f"Aiogram ошибка в handle_grammar: {e}")
        await callback.answer("⚠️ Ошибка при отображении грамматики", show_alert=True)
    except Exception as e:
        logger.error(f"Неизвестная ошибка в handle_grammar: {e}", exc_info=True)
        await callback.answer("⚠️ Произошла внутренняя ошибка", show_alert=True)

# Добавляем глобальный словарь для хранения тематической лексики
theme_classifier = ThemeClassifier()
themes_dict = defaultdict(list)

@dp.callback_query(F.data == CALLBACK_LEXICON)
async def handle_lexicon(callback: types.CallbackQuery):
    """Показ лексики, сгруппированной по темам (с проверкой)"""
    global themes_dict
    try:
        themes_dict.clear()
        has_lexicon = False

        for story in tales_data['stories']:
            if (story.get('han_words') and story.get('rus_words') and
                len(story['han_words']) > 0 and len(story['rus_words']) > 0):
                has_lexicon = True
                min_length = min(len(story['han_words']), len(story['rus_words']))
                for i in range(min_length):
                    han_word = story['han_words'][i].strip()
                    rus_word = story['rus_words'][i].strip()
                    theme = theme_classifier.detect_theme(rus_word)
                    themes_dict[theme].append((han_word, rus_word))

        if not has_lexicon:
            await callback.answer("❌ В словаре нет доступной лексики", show_alert=True)
            return

        builder = InlineKeyboardBuilder()
        for theme in sorted(themes_dict.keys()):
            builder.button(text=theme, callback_data=f"lexicon_theme_{theme}")
        builder.button(text="🔙 Назад", callback_data=CALLBACK_BACK_TO_VOCABULARY)
        builder.adjust(2)

        await callback.message.answer(
            "📚 Выбери тематику словаря:",
            reply_markup=builder.as_markup()
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_lexicon: {e}", exc_info=True)
        await callback.answer("⚠️ Ошибка при загрузке словаря", show_alert=True)


@dp.callback_query(F.data.startswith("lexicon_theme_"))
async def handle_lexicon_theme(callback: types.CallbackQuery):
    """Показывает слова по выбранной теме"""
    global themes_dict

    try:
        theme = callback.data.split('_', 2)[2]
        if theme not in themes_dict:
            await callback.answer("Тема не найдена", show_alert=True)
            return

        words = themes_dict[theme]
        word_list = '\n'.join([f"• <b>{han}</b> — {rus}" for han, rus in words])
        message = f"📚 <b>{theme}</b>\n{word_list}"

        builder = InlineKeyboardBuilder()
        builder.button(text="🔙 Назад", callback_data=CALLBACK_LEXICON)

        await callback.message.answer(
            message,
            reply_markup=builder.as_markup()
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_lexicon_theme: {e}", exc_info=True)
        await callback.answer("⚠️ Ошибка при загрузке темы", show_alert=True)














# --- Обработчики алфавита ---
VOWELS = {'А', 'Ӑ', 'И', 'Й', 'О', 'Ө', 'У', 'Ў', 'Ы', 'Э', 'Є', 'Ә'}

@dp.callback_query(F.data == CALLBACK_ALPHABET)
async def handle_alphabet(callback: types.CallbackQuery):
    try:
        await callback.message.answer(
            "🔤 Выбери раздел хантыйского алфавита:\n\n"
    "Здесь ты можешь прослушать правильное произношение каждой буквы и увидеть её печатную версию.",
    reply_markup=await alphabet_menu_kb()
)
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке меню", show_alert=True)

async def alphabet_menu_kb() -> InlineKeyboardMarkup:
    buttons = [
        ("🔠 Все буквы", CALLBACK_ALPHABET_LETTERS_LIST),
        ("🔡 Гласные", CALLBACK_ALPHABET_VOWELS),
        ("🔣 Согласные", CALLBACK_ALPHABET_CONSONANTS)
    ]
    return build_menu(buttons, ("🔙 Назад", CALLBACK_BACK_TO_VOCABULARY), columns=1)

@dp.callback_query(F.data == CALLBACK_ALPHABET_LETTERS_LIST)
async def handle_alphabet_letters_list(callback: types.CallbackQuery):
    try:
        with open('alphabet.json', 'r', encoding='utf-8') as f:
            alphabet_data = json.load(f)
        
        buttons = []
        for letter in alphabet_data:
            letter_char = Path(letter['photo']).stem
            callback_data = f"{CALLBACK_ALPHABET_LETTER_DETAIL}{letter['name']}"
            buttons.append((letter_char, callback_data))
        
        await callback.message.answer(
            "Все буквы алфавита:",
            reply_markup=build_menu(
                buttons,
                back_button=("🔙 Назад", CALLBACK_ALPHABET),
                columns=4
            )
        )
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке списка букв", show_alert=True)

@dp.callback_query(F.data.startswith(CALLBACK_ALPHABET_LETTER_DETAIL))
async def handle_letter_detail(callback: types.CallbackQuery):
    try:
        letter_name = callback.data.replace(CALLBACK_ALPHABET_LETTER_DETAIL, "")
        
        with open('alphabet.json', 'r', encoding='utf-8') as f:
            alphabet_data = json.load(f)
        
        letter = next((item for item in alphabet_data if item['name'] == letter_name), None)
        
        if not letter:
            await callback.answer("❌ Буква не найдена", show_alert=True)
            return
        
        # Отправляем фото буквы
        photo_path = Path(__file__).parent / letter['photo']
        if photo_path.exists():
            photo = types.FSInputFile(photo_path)
            await callback.message.answer_photo(
                photo,
                caption=f"{letter['name']}\n\nНажми на аудио ниже, чтобы прослушать произношение"
            )
        else:
            await callback.message.answer(f"⚠️ Изображение для {letter['name']} не найдено")
        
        # Отправляем аудио с произношением
        audio_path = Path(__file__).parent / letter['sound']
        if audio_path.exists():
            audio = types.FSInputFile(audio_path)
            await callback.message.answer_audio(audio)
        else:
            await callback.message.answer(f"⚠️ Аудио для {letter['name']} не найдено")
        
        # Определяем откуда пришли
        letter_char = Path(letter['photo']).stem.upper()
        back_callback = CALLBACK_ALPHABET_LETTERS_LIST
        if letter_char in VOWELS:
            back_callback = CALLBACK_ALPHABET_VOWELS
        else:
            back_callback = CALLBACK_ALPHABET_CONSONANTS
        
        await callback.message.answer(
            "Выбери действие:",
            reply_markup=build_menu([], (("🔙 Назад", back_callback)))
        )
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке информации о букве", show_alert=True)

@dp.callback_query(F.data == CALLBACK_ALPHABET_VOWELS)
async def handle_alphabet_vowels(callback: types.CallbackQuery):
    try:
        with open('alphabet.json', 'r', encoding='utf-8') as f:
            alphabet_data = json.load(f)
        
        buttons = []
        for letter in alphabet_data:
            letter_char = Path(letter['photo']).stem.upper()
            if letter_char in VOWELS:
                callback_data = f"{CALLBACK_ALPHABET_LETTER_DETAIL}{letter['name']}"
                buttons.append((letter_char, callback_data))
        
        # Сортируем по порядку гласных
        buttons.sort(key=lambda x: sorted(VOWELS).index(x[0].upper()) if x[0].upper() in VOWELS else len(VOWELS))
        
        await callback.message.answer(
            "🔤 <b>Гласные буквы хантыйского алфавита</b>\n\n"
            "Выбери гласную, чтобы увидеть её написание и услышать произношение.\n\n"
            "ℹ️ Для подробной информации о каждой букве и тонкостях произношения — нажми «📝 Описание гласных».\n\n"
            "⬅️ Чтобы вернуться к меню алфавита, используй кнопку «🔙 Назад».",
            reply_markup=build_menu(
                buttons,
                additional_buttons=[("📝 Описание гласных", CALLBACK_VOWELS_DESCRIPTION)],
                back_button=("🔙 Назад", CALLBACK_ALPHABET),
                columns=4
            )
        )
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке гласных букв", show_alert=True)

@dp.callback_query(F.data == CALLBACK_ALPHABET_CONSONANTS)
async def handle_alphabet_consonants(callback: types.CallbackQuery):
    try:
        with open('alphabet.json', 'r', encoding='utf-8') as f:
            alphabet_data = json.load(f)
        
        buttons = []
        for letter in alphabet_data:
            letter_char = Path(letter['photo']).stem.upper()
            if letter_char not in VOWELS:
                callback_data = f"{CALLBACK_ALPHABET_LETTER_DETAIL}{letter['name']}"
                buttons.append((letter_char, callback_data))
        
        await callback.message.answer(
            "🔤 <b>Согласные буквы хантыйского алфавита</b>\n\n"
            "Выбери согласную, чтобы увидеть её написание и услышать произношение.\n\n"
            "ℹ️ Для подробной информации о каждой букве и тонкостях произношения — нажми «📝 Описание согласных».\n\n"
            "⬅️ Чтобы вернуться к меню алфавита, используй кнопку «🔙 Назад».",
            reply_markup=build_menu(
                buttons,
                additional_buttons=[("📝 Описание согласных", CALLBACK_CONSONANTS_DESCRIPTION)],
                back_button=("🔙 Назад", CALLBACK_ALPHABET),
                columns=4
            )
        )
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке согласных букв", show_alert=True)

@dp.callback_query(F.data == CALLBACK_VOWELS_DESCRIPTION)
async def handle_vowels_description(callback: types.CallbackQuery):
    try:
        with open('phonetics.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        await callback.message.answer(
            data["гласные"],
            reply_markup=build_menu([], ("🔙 Назад", CALLBACK_ALPHABET_VOWELS))
        )
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке описания", show_alert=True)

@dp.callback_query(F.data == CALLBACK_CONSONANTS_DESCRIPTION)
async def handle_consonants_description(callback: types.CallbackQuery):
    try:
        with open('phonetics.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        await callback.message.answer(
            data["согласные"],
            reply_markup=build_menu([], ("🔙 Назад", CALLBACK_ALPHABET_CONSONANTS))
        )
        await callback.answer()
    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке описания", show_alert=True)











# Включаем возможность загрузки усечённых изображений (временное решение)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Глобальный кэш изображений
image_cache: Dict[str, bytes] = {}



async def compress_image(image_path: Path, quality=75) -> bytes:
    """Сжимает изображение с проверкой целостности и более агрессивной оптимизацией"""
    try:
        with Image.open(image_path) as img:
            # Быстрая проверка целостности
            img.verify()
            
        with Image.open(image_path) as img:
            # Конвертируем в RGB и уменьшаем размер
            img = img.convert("RGB")
            
            # Определяем оптимальный размер для Telegram (до 1280px по большей стороне)
            max_size = 1280
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)
                
            buffer = io.BytesIO()
            
            # Более агрессивная оптимизация
            img.save(
                buffer, 
                format="JPEG", 
                quality=quality, 
                optimize=True, 
                progressive=True
            )
            
            return buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Ошибка при обработке изображения {image_path.name}: {str(e)}")


async def preload_images():
    """Предзагружает и сжимает все изображения при старте с обработкой ошибок"""
    loaded_count = 0
    for story in tales_data['stories']:
        illustr_dir = Path(__file__).parent / "illustraciones" / story['rus_title']
        if not illustr_dir.exists():
            continue
            
        try:
            images = sorted(
                [img for img in illustr_dir.iterdir() 
                 if img.is_file() and img.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                key=lambda x: x.name
            )
            
            for img in images:
                try:
                    # Сжимаем с более агрессивными настройками
                    image_cache[str(img)] = await compress_image(img, quality=75)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Не удалось загрузить {img.name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке папки {illustr_dir}: {str(e)}")
    
    logger.info(f"Успешно предзагружено {loaded_count} изображений")


async def send_multiple_photos(chat_id: int, photos: List[Path]):
    """Отправляет несколько фото параллельно"""
    tasks = []
    for photo in photos:
        if str(photo) in image_cache:
            tasks.append(
                bot.send_photo(
                    chat_id=chat_id,
                    photo=types.BufferedInputFile(image_cache[str(photo)], filename=photo.name),
                    caption=f"Иллюстрация {photos.index(photo)+1}/{len(photos)}"
                )
            )
    await asyncio.gather(*tasks)


def get_story_images(story: dict) -> list:
    """Возвращает список изображений для сказки"""
    illustr_dir = Path(__file__).parent / "illustraciones" / story['rus_title']
    if not illustr_dir.exists():
        return []
    return sorted(
        [img for img in illustr_dir.iterdir() if img.is_file() and img.suffix.lower() in ('.jpg', '.jpeg', '.png')],
        key=lambda x: x.name
    )


@dp.callback_query(F.data.startswith(CALLBACK_SHOW_ILLUSTRATIONS))
async def handle_show_illustrations(callback: CallbackQuery, state: FSMContext):
    """Показ иллюстраций к сказке"""
    try:
        story_id = int(callback.data.replace(CALLBACK_SHOW_ILLUSTRATIONS, ""))
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        images = get_story_images(story)

        if not images:
            await callback.answer("❌ Иллюстрации не найдены", show_alert=True)
            return

        await send_illustration_page(callback.message, story, images, 0, state)
        await callback.answer()

    except Exception as e:
        logger.error(f"Ошибка в handle_show_illustrations: {e}")
        await callback.answer("⚠️ Ошибка при загрузке иллюстраций", show_alert=True)


async def send_illustration_page(message: Message, story: dict, images: list, page: int, state: FSMContext):
    """Отправляет одну иллюстрацию с навигацией"""
    try:
        if page < 0 or page >= len(images):
            raise IndexError("Некорректный номер страницы")
            
        image_path = images[page]
        caption = f"🖼️ Иллюстрация {page+1}/{len(images)}\n<b>{story['rus_title']}</b>"

        # Получаем сжатое изображение
        if str(image_path) not in image_cache:
            image_cache[str(image_path)] = await compress_image(image_path)

        # Получаем сохраненный язык
        user_data = await state.get_data()
        lang = user_data.get('last_lang', 'ru')
        back_callback = f"{CALLBACK_LANGUAGE_RU}{story['id']}" if lang == 'ru' else f"{CALLBACK_LANGUAGE_KH}{story['id']}"

        # Создаем клавиатуру
        builder = InlineKeyboardBuilder()
        
        if page > 0:
            builder.button(text="◀️ Назад", callback_data=f"illustr_prev_{story['id']}_{page}")
        if page < len(images) - 1:
            builder.button(text="Вперед ▶️", callback_data=f"illustr_next_{story['id']}_{page}")
            
        builder.button(text="🔙 Назад к сказке", callback_data=back_callback)
        builder.adjust(2)
        
        # Отправляем фото
        await message.answer_photo(
            types.BufferedInputFile(
                image_cache[str(image_path)],
                filename=f"illustration_{page}.jpg"
            ),
            caption=caption,
            reply_markup=builder.as_markup()
        )

    except Exception as e:
        logger.error(f"Ошибка при отправке иллюстрации: {e}")
        raise


@dp.callback_query(F.data.startswith("illustr_prev_"))
async def handle_illustr_prev(callback: CallbackQuery, state: FSMContext):
    """Переход к предыдущей иллюстрации"""
    try:
        parts = callback.data.split('_')
        story_id = int(parts[2])
        current_page = int(parts[3])

        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        images = get_story_images(story)

        await callback.message.delete()
        await send_illustration_page(callback.message, story, images, current_page - 1, state)
        await callback.answer()

    except Exception as e:
        logger.error(f"Ошибка в handle_illustr_prev: {e}")
        await callback.answer("⚠️ Ошибка при переходе", show_alert=True)


@dp.callback_query(F.data.startswith("illustr_next_"))
async def handle_illustr_next(callback: CallbackQuery, state: FSMContext):
    """Переход к следующей иллюстрации"""
    try:
        parts = callback.data.split('_')
        story_id = int(parts[2])
        current_page = int(parts[3])

        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        images = get_story_images(story)

        await callback.message.delete()
        await send_illustration_page(callback.message, story, images, current_page + 1, state)
        await callback.answer()

    except Exception as e:
        logger.error(f"Ошибка в handle_illustr_next: {e}")
        await callback.answer("⚠️ Ошибка при переходе", show_alert=True)

# --- Обработчики навигации ---
@dp.callback_query(F.data == CALLBACK_BACK_TO_MAIN)
async def handle_back_to_main(callback: types.CallbackQuery):
    """Возврат в главное меню"""
    try:
        await callback.message.answer(
            f"🌟 <b>{html.escape(callback.from_user.first_name)}</b>, ты в главном меню! \n \n"
            "Выбери <b>📖 Cказки</b>, если хочешь: \n"
            " • почитать или послушать сказки на хантыйском,\n"
            " • увидеть русский перевод сказки,\n"
            " • пройти тест на знание материала,\n\n"
             
            "Выбери <b>📚 Словарик</b>, если хочешь:\n"
            " • услышать произношение букв хантыйского алфавита,\n"
            " • увидеть список слов с переводом,\n"
            " • узнать грамматические правила. \n\n",
            
            reply_markup=await main_menu_kb(),
            parse_mode=ParseMode.HTML
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_back_to_main: {e}")
        await callback.answer("⚠️ Ошибка при возврате в меню", show_alert=True)


@dp.callback_query(F.data == CALLBACK_BACK_TO_TALES)
async def handle_back_to_tales(callback: types.CallbackQuery):
    """Возврат в меню сказок"""
    try:
        await callback.message.answer(
            "📖 Выбери сказку или воспользуйся кнопками <b>Вперёд ▶️</b> и <b>◀️ Назад</b> для перехода по меню:",
            reply_markup=await tales_menu_kb()
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_back_to_tales: {e}")
        await callback.answer("⚠️ Ошибка при возврате в меню", show_alert=True)


@dp.callback_query(F.data == CALLBACK_BACK_TO_VOCABULARY)
async def handle_back_to_vocabulary(callback: types.CallbackQuery):
    """Возврат в меню словаря"""
    try:
        await callback.message.answer(
           "📚 Выбери раздел словаря:\n\n"
            "В <b>📝 Общей грамматике</b> можешь прочитать о грамматических правилах: \n" 
            " • Сколько чисел в хантыйском и как они образуются,\n "
            " • Какие есть падежные суффиксы,\n"
            " • Как ласково сказать белочка или рыбка.\n\n"

            "В <b>🔤 Общей лексике</b> сможешь узнать слова из разных категорий:\n" 
            " • Еда,\n" 
            " • Животные,\n"
            " • Природа и другие.\n\n"

            "В <b>🔡 Алфавите</b> можешь увидеть:\n" 
            " • Названия букв\n" 
            " • Гласные звуки\n" 
            " • Согласные звуки.\n"
            ,
            reply_markup=await vocabulary_menu_kb()
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в handle_back_to_vocabulary: {e}")
        await callback.answer("⚠️ Ошибка при возврате в меню", show_alert=True)











# --- Обработчики текстовых сообщений ---
@dp.message(F.text)
async def handle_text(message: types.Message):
    """Обработчик текстовых сообщений"""
    await message.answer("Пожалуйста, используйте кнопки меню или команду /start")


@dp.message()
async def handle_other(message: types.Message):
    """Обработчик всех необработанных сообщений"""
    await message.answer("Извините, я не понимаю этот тип сообщений. Используйте кнопки меню.")


# --- Глобальный обработчик ошибок ---
@dp.error()
async def errors_handler(event: types.ErrorEvent):
    """Глобальный обработчик ошибок для aiogram 3.x"""
    logger.error(
        "Ошибка при обработке события %s: %s",
        event.update,
        event.exception,
        exc_info=True
    )
    return True

# --- Запуск бота ---
async def main():
    try:
        logger.info("Запуск бота...")
        await set_bot_commands(bot)  # Добавьте эту строку
        await preload_images()  # Добавьте эту строку перед start_polling
        await dp.start_polling(bot)
    except Exception as e:
        logger.critical(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()
        logger.info("Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())



# версия 14 августа















