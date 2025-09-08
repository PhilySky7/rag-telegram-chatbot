# bot.py

import os
import asyncio
import logging
from typing import Dict, Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message

from dotenv import load_dotenv
from rag import RAGPipeline
from messages import MESSAGE

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
HF_KEY = os.getenv("HF_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# log = logging.getLogger(__name__)

router = Router()

# ===== FSM =====
class Keys(StatesGroup):
    wait_pplx = State()
    wait_hf = State()


class Build(StatesGroup):
    wait_pdf = State()
    ready = State()

# ===== In-memory per-user storage (для демо) =====
user_pplx: Dict[int, str] = {}
user_hf: Dict[int, Optional[str]] = {}
user_prompt: Dict[int, Optional[str]] = {}
user_pipeline: Dict[int, Optional[RAGPipeline]] = {}

DATA_DIR = os.path.join(os.getcwd(), "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

def persist_dir_for(user_id: int) -> str:
    d = os.path.join(CHROMA_DIR, str(user_id))
    os.makedirs(d, exist_ok=True)
    return d

# ===== Commands =====
@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Initial message."""
    
    logging.info("Bot started")
    await state.clear()
    await message.answer(MESSAGE['initial_message'])
    await state.set_state(Build.wait_pdf)
    
    
@router.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext):
    await message.answer(MESSAGE["help_message"])
    await state.set_state(Build.wait_pdf)


@router.message(Command("info"))
async def cmd_info(message: Message, state: FSMContext):
    await message.answer(MESSAGE["info"])
    await state.set_state(Build.wait_pdf)


# ===== Upload flow =====

@router.message(Build.wait_pdf, F.document)
async def on_pdf(message: Message, state: FSMContext):
    if not message.document.file_name.lower().endswith(".pdf"):
        await message.answer("Нужен PDF-файл. Пришлите документ с расширением .pdf")

    # Save file
    file_info = await message.bot.get_file(message.document.file_id)
    logging.info("Document is loading")
    await message.answer("Документ загружается, ожидай.")
    
    dest = os.path.join(DOCS_DIR, f"{message.document.file_name}")
    await message.bot.download_file(
        file_info.file_path,
        destination=dest,
    )
    logging.info("Document is loaded")

    # assemble RAG
    pipe = RAGPipeline(
        pplx_api_key=API_KEY,
        hf_api_key=HF_KEY,
        prefer_endpoint=True,  # if timeout risks switch to local
        persist_directory=persist_dir_for(message.from_user.id),
        collection_name=f"user_{message.from_user.id}",
    )
    try:
        pipe.index_pdf(dest)
        user_pipeline[message.from_user.id] = pipe
        await message.answer("Документ загружен и проиндексирован. Теперь можно задавать вопросы.")
        await state.set_state(Build.ready)
    except Exception as e:
        logging.exception("Index error: %s", e)
        await message.answer("Не удалось построить RAG по документу.")

# ===== Q&A after ready =====
@router.message(Build.ready, F.text)
async def on_query(message: Message, state: FSMContext):
    pipe = user_pipeline.get(message.from_user.id)
    if not pipe:
        await message.answer("Сначала загрузите документ.")
        await state.set_state(Build.wait_pdf)
        return
    question = (message.text or "").strip()
    if not question:
        await message.answer("Отправьте текстовый вопрос.")
        return
    try:
        # answer, _ = pipe.ask(question)
        answer = pipe.ask(question)
        # src = "\n".join(f"{i+1}. {s}" for i, s in enumerate(pipe.format_sources(ctx))) or "—"
        await message.answer(
            f"Ответ:\n\n{answer}\n\n"
        )
    except Exception as e:
        logging.exception("Ask error: %s", e)
        await message.answer("Ошибка при обработке вопроса. Попробуй загрузить файл заново.")

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
