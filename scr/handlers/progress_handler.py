 # src/handlers/progress_handler.py
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Union

from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.core.config import logger, CALLBACK_BACK_TO_MAIN, tales_data, CALLBACK_PROGRESS
from src.db.database import Database



router = Router()

@router.callback_query(F.data == CALLBACK_PROGRESS)
@router.message(Command("progress"))
async def cmd_progress(update: Union[types.Message, types.CallbackQuery], db: Database):
    """Универсальный обработчик прогресса"""
    try:
        if isinstance(update, types.CallbackQuery):
            message = update.message
            user = update.from_user
            is_callback = True
        else:
            message = update
            user = update.from_user
            is_callback = False
        
        progress = db.get_user_progress(user.id)
        
        tales_read = progress["tales_read"]
        total_reads = progress["total_reads"]
        tales_completed = progress["tales_completed"]
        recent_tales = progress["recent_tales"]
        
        progress_text = (
            f"📊 <b>Ваш прогресс:</b>\n"
            f" •📜 Прочитано сказок: <b>{tales_read}</b>\n"
            f" •🔁 Всего прочтений: <b>{total_reads}</b>\n"
            f" •🏁 Завершено тестов: <b>{tales_completed}</b>\n\n"
        )
        
        
        # Получаем завершенные сказки (используем метод из Database или напрямую SQL)
        import sqlite3
        from contextlib import closing
        from datetime import datetime
        
        with closing(sqlite3.connect(db.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
                SELECT tale_id, read_count 
                FROM tale_progress 
                WHERE user_id = ? AND completed = TRUE 
                ORDER BY last_read_date DESC 
            """, (user.id,))
            completed_tales = cursor.fetchall()
        
        if completed_tales:
            progress_text += "<b>✅ Завершённые сказки:</b>\n"
            for tale_id, read_count in completed_tales:
                story = next((s for s in tales_data['stories'] if s['id'] == tale_id), None)
                if story:
                    progress_text += f" •🗞️ <b>{story['rus_title']}</b> (прочитано {read_count} раз(а))\n"
            progress_text += "\n\n"
        
        # Добавляем информацию о недавно прочитанных сказках
        if recent_tales:
            progress_text += "<b>📚 Недавно прочитанные:</b>\n"
            for tale in recent_tales:
                tale_id, last_read, read_count, completed = tale
                story = next((s for s in tales_data['stories'] if s['id'] == tale_id), None)
                if story:
                    status = "📗" if completed else "📖"
                    
                    # Форматируем дату в дд.мм.гггг
                    try:
                        if last_read:
                            # Пытаемся распарсить дату из ISO формата
                            date_obj = datetime.fromisoformat(last_read)
                            last_read_date = date_obj.strftime("%d.%m.%Y")
                        else:
                            last_read_date = "неизвестно"
                    except (ValueError, TypeError):
                        last_read_date = "неизвестно"
                    
                    progress_text += (
                        f" •{status} <b>{story['rus_title']}</b> "
                        f"(прочитано {read_count} раз(а), последнее прочтение {last_read_date})"
                        "\n"
                    )
        
        # Создаем клавиатуру для возврата в главное меню
        from aiogram.utils.keyboard import InlineKeyboardBuilder
        
        builder = InlineKeyboardBuilder()
        builder.button(text="🗂️ Главное меню", callback_data=CALLBACK_BACK_TO_MAIN)
        
        if is_callback:
            await message.answer(progress_text, reply_markup=builder.as_markup(), parse_mode=ParseMode.HTML)
            await update.answer()
        else:
            await message.answer(progress_text, reply_markup=builder.as_markup(), parse_mode=ParseMode.HTML)
            
    except Exception as e:
        logger.error(f"Ошибка в cmd_progress: {e}")
        error_msg = "⚠️ Ошибка при загрузке прогресса"
        
        if isinstance(update, types.CallbackQuery):
            await update.answer(error_msg, show_alert=True)
        else:
            await update.answer(error_msg)



            