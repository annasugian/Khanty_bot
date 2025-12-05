# src/core/config.py

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
from collections import defaultdict
import re

# --- Настройка логов ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

themes_dict: Dict[str, List[Tuple[str, str]]] = {}

# --- Явная загрузка .env ---
# Путь к .env теперь должен быть относительно корня проекта
env_path = Path(__file__).parent.parent.parent / '.env'
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
CALLBACK_VOWELS_DESCRIPTION = "vowels_description"
CALLBACK_CONSONANTS_DESCRIPTION = "consonants_description"
CALLBACK_ALPHABET_DESCRIPTION = "alphabet_description"
CALLBACK_SHOW_ILLUSTRATIONS = "show_illustrations_"
CALLBACK_PROGRESS = "show_progress"
CALLBACK_SHOW_CULTURE = "show_culture_"
CALLBACK_ALPHABET_LETTERS_LIST = "alphabet_letters_list"



# --- Функции загрузки данных ---
# Пути к файлам данных должны быть относительно корня проекта
BASE_DATA_PATH = Path(__file__).parent.parent.parent

def load_tales_from_json(json_path: str) -> dict:
    try:
        with open(BASE_DATA_PATH / json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Успешно загружено {len(data['stories'])} сказок из JSON")
            return data
    except Exception as e:
        logger.error(f"Ошибка загрузки JSON: {e}")
        raise

def load_tests_from_json(json_path: str) -> dict:
    try:
        with open(BASE_DATA_PATH / json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Успешно загружено {len(data['tests'])} тестов из JSON")
            return data
    except Exception as e:
        logger.error(f"Ошибка загрузки тестов JSON: {e}")
        raise

def load_phonetics():
    try:
        with open(BASE_DATA_PATH / "phonetics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки phonetics.json: {e}")
        return None

CULTURE_FILE = BASE_DATA_PATH / 'culture.json'

def load_culture_data():
    """Загрузка данных из culture.json с учётом структуры {'culture': [...]}"""
    try:
        if not CULTURE_FILE.exists():
            logger.error(f"Файл не найден: {CULTURE_FILE}")
            return []

        with open(CULTURE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'culture' in data:
                logger.info("Обнаружена структура с ключом 'culture'")
                return [
                    item for item in data['culture'] 
                    if isinstance(item, dict) 
                    and item.get('fact', '').strip()
                ]
            
            logger.error("Неверная структура файла culture.json")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Критическая ошибка загрузки: {e}")
        return []

# --- Загрузка данных (Глобальные переменные) ---
try:
    tales_data = load_tales_from_json("fairytales.json")
except Exception as e:
    logger.critical(f"Не удалось загрузить данные сказок: {e}")
    exit(1)

try:
    tests_data = load_tests_from_json("tests.json")
except Exception as e:
    logger.error(f"Не удалось загрузить тесты: {e}")
    tests_data = {"tests": []}

phonetics_data = load_phonetics()
culture_data = load_culture_data()
logger.info(f"Загружено культурных фактов: {len(culture_data)}")


KHANTY_ALPHABET_ORDER = [
    'а', 'ӑ', 'в', 'и', 'й', 
    'к', 'л', 'љ', 'ԓ', 'м', 'н', 
    'њ', 'ӈ', 'о', 'ө', 'п', 'р', 
    'с', 'т', 'Ђ', 'у', 'ў', 'х',
    'ш', 'щ', 'ы', 'э', 'є', 'ә'
]

# Словарь для быстрого поиска порядка
KHANTY_ORDER_DICT = {letter: idx for idx, letter in enumerate(KHANTY_ALPHABET_ORDER)}

def khanty_sort_key(word_pair: tuple) -> list:
    """Ключ сортировки для хантыйских слов."""
    if not word_pair or not word_pair[0]:
        return [9999]
    
    han_word = word_pair[0].lower()  
    order_values = []
    
    for char in han_word:
        if char in KHANTY_ORDER_DICT:
            order_values.append(KHANTY_ORDER_DICT[char])
        else:
            order_values.append(ord(char) + 10000)
    
    return order_values


def sort_khanty_words_in_themes(themes_dict: dict) -> dict:
    """
    Сортирует слова внутри тем по хантыйскому алфавиту.
    
    Args:
        themes_dict: словарь тем {theme: [(han_word, rus_word), ...]}
    
    Returns:
        dict: отсортированный словарь тем
    """
    if not themes_dict:
        return themes_dict
    
    sorted_themes_dict = {}
    for theme, word_pairs in themes_dict.items():
        if word_pairs:
            # Сортируем по хантыйскому слову
            sorted_pairs = sorted(word_pairs, key=khanty_sort_key)
            sorted_themes_dict[theme] = sorted_pairs
        else:
            sorted_themes_dict[theme] = []
    
    return sorted_themes_dict

# Альтернативная функция для сортировки по русскому переводу (fallback)
def sort_by_russian_translation(themes_dict: dict) -> dict:
    """Сортирует слова внутри тем по русскому переводу."""
    if not themes_dict:
        return themes_dict
    
    sorted_themes_dict = {}
    for theme, word_pairs in themes_dict.items():
        if word_pairs:
            sorted_pairs = sorted(word_pairs, key=lambda x: x[1].lower())
            sorted_themes_dict[theme] = sorted_pairs
        else:
            sorted_themes_dict[theme] = []
    
    return sorted_themes_dict

PHOTO_LICENSES = {
    0: "Фото: Wikimedia Commons (CC BY-SA 3.0)",
    1: "Фото: A.Savin, Wikimedia Commons (CC BY-SA 4.0)",
    2: " ",  
    3: "Фото: ttelegraf.ru",
    4: "Фото: Wikimedia Commons (CC BY-SA 3.0)",  
    5: " ",  
    6: "Фото: Красная книга ХМАО",
    7: "Фото: animalzoom.ru",
    8: "Фото: etosibir.ru",
    9: "Фото: torummaa.ru",
    10: "Фото: Яндекс.Фотки",
    11: "Фото: ugra-tv.ru",
    12: "Фото: A.Savin, Wikimedia Commons (CC BY-SA 4.0)",  
    13: "Фото: informugra.ru",
    15: "Фото: geoglob.ru",
    16: " ",  
    17: " ",  
    18: "Фото: ljplus.ru"
}
