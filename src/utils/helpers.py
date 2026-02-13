# src/utils/helpers.py

import asyncio
import io
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageFile
from aiogram import types, Bot
from aiogram.types import BufferedInputFile, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
import os
from aiogram.fsm.context import FSMContext

# Импорт из модулей проекта
from src.core.config import logger, tales_data, CALLBACK_LANGUAGE_RU, CALLBACK_LANGUAGE_KH 

# Включаем возможность загрузки усечённых изображений (временное решение)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Глобальный кэш изображений 
image_cache: Dict[str, bytes] = {} 

# Путь к корню проекта для файлов
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Вспомогательные функции ---
# src/utils/helpers.py

async def split_long_message(text: str, max_length: int = 4096) -> List[str]:
    """Разбивает длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]
    
    parts = []
    while text:
        # Берем часть текста не больше max_length
        part = text[:max_length]
        
        # Ищем последний перенос строки в этой части
        split_pos = part.rfind('\n') if '\n' in part else max_length
        
        # Добавляем часть до переноса строки
        parts.append(text[:split_pos])
        
        # Обрезаем обработанную часть текста
        text = text[split_pos:].lstrip()
    
    return parts


async def send_audio_if_exists(bot: Bot, chat_id: int, story: dict):
    """Отправляет аудиофайл, если он существует"""
    if story.get('audio') and story['audio'] != "pass":
        audio_path = BASE_DIR / "audio" / story['audio']
        try:
            if audio_path.exists():
                audio_file = FSInputFile(audio_path)
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


async def compress_image(image_path: Path, quality: int = 75) -> bytes:
    """Сжимает изображение с проверкой целостности и более агрессивной оптимизацией"""
    try:
        with Image.open(image_path) as img:
            img.verify() # Быстрая проверка целостности

        with Image.open(image_path) as img:
            img = img.convert("RGB") # Конвертируем в RGB
            
            # Определяем оптимальный размер для Telegram (до 1280px по большей стороне)
            max_size = 1280
            if max(img.size) > max_size:
                # Используем Image.Resampling.LANCZOS в современных PIL
                try:
                    resample_method = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_method = Image.LANCZOS # Для старых версий
                img.thumbnail((max_size, max_size), resample_method)
                
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

async def preload_images(bot: Bot):
    """Предзагружает и сжимает все изображения при старте с обработкой ошибок"""
    global image_cache
    
    # Очищаем кэш перед загрузкой
    image_cache = {} 
    loaded_count = 0
    
    ILLUSTRATIONS_ROOT = BASE_DIR / "illustraciones"
    
    for story in tales_data['stories']:
        illustr_dir = ILLUSTRATIONS_ROOT / story['rus_title']
        if not illustr_dir.exists():
            continue
            
        try:
            # Получаем все изображения, сортируем по имени
            images = sorted(
                [img for img in illustr_dir.iterdir() if img.is_file() and img.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                key=lambda x: x.name
            )
            
            for img in images:
                try:
                    # Сжимаем и кэшируем
                    image_cache[str(img)] = await compress_image(img, quality=75)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Не удалось загрузить/сжать {img.name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке папки {illustr_dir.name}: {e}")
            
    logger.info(f"Успешно предзагружено и кэшировано {loaded_count} изображений.")


async def show_illustration(message: types.Message, story_id: int, page: int, state: FSMContext):
    """Показывает одну иллюстрацию с навигацией"""
    try:
        story = next(s for s in tales_data['stories'] if s['id'] == story_id)
        
        illustr_dir = BASE_DIR / "illustraciones" / story['rus_title']
        
        images = sorted(
            [img for img in illustr_dir.iterdir() if img.is_file() and img.suffix.lower() in ('.jpg', '.jpeg', '.png')],
            key=lambda x: x.name
        )
        
        if not images:
             raise ValueError("Иллюстрации не найдены в папке")
             
        if page < 0 or page >= len(images):
            # Если страница некорректна, показываем первую
            page = 0 
        
        image_path = images[page]
        caption = f"🖼️ Иллюстрация {page+1}/{len(images)}\n<b>{story['rus_title']}</b>"
        
        # Получаем сжатое изображение из кэша
        image_key = str(image_path)
        if image_key not in image_cache:
            # Если по какой-то причине не в кэше, сжимаем
            image_cache[image_key] = await compress_image(image_path)
        
        # Получаем сохраненный язык для кнопки "Назад к сказке"
        user_data = await state.get_data()
        lang = user_data.get('last_lang', 'ru')
        back_callback = f"{CALLBACK_LANGUAGE_RU}{story['id']}" if lang == 'ru' else f"{CALLBACK_LANGUAGE_KH}{story['id']}"
        
        # Создаем клавиатуру
        builder = InlineKeyboardBuilder()
        if page > 0:
            builder.button(text="◀️ Назад", callback_data=f"illustr_prev_{story['id']}_{page}")
        if page < len(images) - 1:
            builder.button(text="Вперёд ▶️", callback_data=f"illustr_next_{story['id']}_{page}")
            
        builder.button(text="🔙 Назад к сказке", callback_data=back_callback)
        builder.adjust(2)

        # Отправляем фото
        await message.answer_photo(
            BufferedInputFile(
                image_cache[image_key], 
                filename=f"illustration_{page}.jpg"
            ),
            caption=caption,
            reply_markup=builder.as_markup()
        )
        
    except Exception as e:
        logger.error(f"Ошибка при отправке иллюстрации: {e}")
        raise
