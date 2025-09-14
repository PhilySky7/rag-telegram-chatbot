import os
import re
import time
import logging
import asyncio
from typing import Dict, Optional
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from dotenv import load_dotenv

from rag import build_qa_chain, ask
from messages import MESSAGE
from config import settings


# Central logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_bot")

load_dotenv()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# General constants
MAX_FILE_MB = 20  # Telegram bots usually allow up to ~20 MB
FILES_KEEP_PER_USER = 5  # rotation per user

DATA_DIR = os.path.join(os.getcwd(), "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

router = Router()

# FSM conditions
class Build(StatesGroup):
    wait_pdf = State()
    ready = State()

# In-memory per-user storage
user_pipeline: Dict[int, Optional[object]] = {}

# Additional functions
def sanitize_filename(filename: str) -> str:
    # Оставляем только буквы, цифры, тире, подчёркивание и точку
    return re.sub(r"[^a-zA-Zа-яА-Я0-9._-]", "_", os.path.basename(filename))

def allowed_file(filename: str) -> bool:
    return filename.lower().endswith(".pdf")

def file_too_large(file_path: str, limit_mb: int = MAX_FILE_MB) -> bool:
    return os.path.getsize(file_path) > limit_mb * 1024 * 1024

def cleanup_old_files(dir_path: str, files_keep: int = FILES_KEEP_PER_USER):
    """Простой ротационный механизм: оставляет только N свежих файлов"""
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    files = sorted(files, key=os.path.getmtime, reverse=True)
    for f in files[files_keep:]:
        try: 
            os.remove(f)
        except Exception: 
            pass

# Commands
@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(MESSAGE["initial_message"])
    await state.set_state(Build.wait_pdf)

@router.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext):
    await message.answer(MESSAGE["help_message"])
    await state.set_state(Build.wait_pdf)

@router.message(Command("info"))
async def cmd_info(message: Message, state: FSMContext):
    await message.answer(MESSAGE["info"])
    await state.set_state(Build.wait_pdf)

# Download PDF
@router.message(Build.wait_pdf, F.document)
async def on_pdf(message: Message, state: FSMContext):
    filename = sanitize_filename(message.document.file_name)
    if not allowed_file(filename):
        await message.answer("Нужен PDF-файл. Пришлите документ с расширением .pdf.")
        return

    cleanup_old_files(DOCS_DIR)
    
    # Download the file and check its size
    file_info = await message.bot.get_file(message.document.file_id)
    dest = os.path.join(DOCS_DIR, filename)
    await message.bot.download_file(file_info.file_path, destination=dest)
    if file_too_large(dest):
        os.remove(dest)
        logger.warning("File size is too large")
        await message.answer("PDF слишком большой (лимит 20 МБ)")
        return


    await message.answer("Документ загружается и будет проиндексирован, подождите...")

    # Pipeline indexing
    t0 = time.time()
    try:
        api_key = PERPLEXITY_API_KEY
        if not api_key:
            await message.answer("Не задан API ключ для LLM.")
            return
        
        pipeline = await asyncio.to_thread(
            build_qa_chain,
            pdf_path=dest,
            api_key=api_key,
            hf_api_key=HF_API_KEY,
            persist_directory=os.path.join(settings.PERSIST_DIRECTORY, str(message.document.file)), 
            collection_name=f"user_{message.from_user.id}",
            prefer_endpoint=True,
            language="ru",
        )
        idx_time = time.time() - t0
        user_pipeline[message.from_user.id] = pipeline

        text = (
            "Документ проиндексирован ✅\n"
            f"Время индексации: {idx_time:.1f} сек.\n"
            "Теперь можешь задавать вопросы."
        )
        await message.answer(text)
        await state.set_state(Build.ready)
    except Exception as e:
        logger.exception(f"Index error: {e}")
        await message.answer("Не удалось корректно обработать PDF. Попробуйте другой файл!")
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except Exception:
            pass
        

@router.message(Build.ready, F.text)
async def on_query(message: Message, state: FSMContext):
    pipe = user_pipeline.get(message.from_user.id)
    if not pipe:
        await message.answer("Загрузите документ, прежде чем задавать вопросы.")
        await state.set_state(Build.wait_pdf)
        return
    
    question = (message.text or "").strip()
    if not question:
        await message.answer("Отправьте корректный вопрос.")
        return
    
    try:
        answer = await asyncio.to_thread(ask, pipe, question)
        await message.answer(f"Ответ:\n\n{answer}")
    except Exception as e:
        logger.exception(f"Ask error: {e}")
        await message.answer("Ошибка при получении ответа. Попробуйте перезагрузить PDF.")


# ===== Catch: until pdf =====
@router.message(F.text)
async def before_ready_text(message: Message, state: FSMContext):
    cur_state = await state.get_state()
    if cur_state in (None, Build.wait_pdf.state):
        await state.set_state(Build.wait_pdf)
        await message.answer(MESSAGE["info"])


async def main():
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    bot = Bot(token=TG_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
