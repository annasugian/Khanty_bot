#main.py

import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv


# --- Настройка логирования (перед импортами) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Импорты модулей проекта ---
from src.handlers import user_commands, tale_handlers, progress_handler  
from src.utils.helpers import preload_images 

from src.db.database import Database 
from src.core.config import tales_data 
from src.middlewares.dependencies import DependencyMiddleware

# --- Инициализация ---
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Инициализация Database
database = Database('user_progress.db')

# Инициализация бота и диспетчера
bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()


# Применяем middleware ко всем сообщениям
dp.message.outer_middleware(DependencyMiddleware(db=database, tales_data=tales_data))
# Применяем middleware ко всем callback_query
dp.callback_query.outer_middleware(DependencyMiddleware(db=database, tales_data=tales_data))


# Регистрация роутеров из ваших модулей
dp.include_routers(
    user_commands.router,
    progress_handler.router,
    tale_handlers.router, 
)

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
    return True # Возвращаем True, чтобы aiogram знал, что ошибка обработана

# --- Запуск бота ---
async def main():
    try:
        logger.info("Запуск бота...")
        
        # set_bot_commands вызывается из импортированного модуля (user_commands)
        await user_commands.set_bot_commands(bot)
        
        # preload_images вызывается из импортированного модуля (helpers)
        await preload_images(bot) 
        
        await dp.start_polling(bot)
    except Exception as e:
        logger.critical(f"Ошибка при запуске бота: {e}")
    finally:
        logger.info("Бот остановлен")
        # В aiogram 3.x явное закрытие сессии не требуется, но вы можете добавить:
        # await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем (Ctrl+C)")