from langchain.prompts import PromptTemplate

template_eng = """
You are an AI assistant for answering questions about a document.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I don't know." Don't try to make up an answer.
Context: {context}
=========
Answer:
"""

PROMPT_ENG = PromptTemplate(template=template_eng, input_variables=["context"])

template_ru = """
Ты ИИ ассистент, который ответит на вопросы по документу.
Тебе будут даны части документа и вопрос. Предоставь развернутый ответ.
Если не знаешь ответ, то так и скажи. Не пытайся выдумать ответ.
Контекст: {context}
=========
Ответ:
"""

PROMPT_RU = PromptTemplate(template=template_ru, input_variables=["context"])

