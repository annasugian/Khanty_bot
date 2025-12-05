from typing import Callable, Dict, Any, Awaitable
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

# Import your Database class 
from src.db.database import Database 

class DependencyMiddleware(BaseMiddleware):
    def __init__(self, db: Database, tales_data: dict):
        # Сохраняем объекты, которые мы хотим инжектировать
        self.db = db
        self.tales_data = tales_data
        super().__init__()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:

        data["db"] = self.db
        data["tales_data"] = self.tales_data
        
        # Передаем управление дальше
        return await handler(event, data)