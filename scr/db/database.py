# src/db/database.py

import sqlite3
from contextlib import closing
from datetime import datetime
from aiogram import types 
from typing import Union
from pathlib import Path
import os

# Получаем путь к корню проекта, чтобы сохранить базу данных
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Класс для работы с базой данных ---
class Database:
    def __init__(self, db_name: str = "user_progress.db"):
        self.db_path = BASE_DIR / db_name
        self._init_db()

    def _init_db(self):
        """Инициализация базы данных и создание таблиц"""
        with closing(sqlite3.connect(self.db_path)) as conn:
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
        with closing(sqlite3.connect(self.db_path)) as conn:
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
        with closing(sqlite3.connect(self.db_path)) as conn:
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
        with closing(sqlite3.connect(self.db_path)) as conn:
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
        with closing(sqlite3.connect(self.db_path)) as conn:
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
        with closing(sqlite3.connect(self.db_path)) as conn:
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