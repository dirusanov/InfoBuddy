import json
import os
from typing import List

import phonenumbers
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("API-ключ OpenAI не указан. Убедитесь, что он есть в файле .env.")

JSON_PATH = "products.json"
INDEX_PATH = "faiss_index"
PHONE_LOG_PATH = "phone_log.json"

OpenAIEmbeddings.OPENAI_API_KEY = openai_api_key
embeddings = OpenAIEmbeddings()


def load_json_as_documents(json_path: str) -> List[Document]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []

    def parse_json(data, parent_key: str = ""):
        if isinstance(data, dict):
            # Сохраняем полный контекст для каждого объекта
            full_context = "\n".join([f"{key}: {value}" for key, value in data.items()])
            documents.append(Document(page_content=full_context))

            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                parse_json(value, new_key)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                parse_json(item, f"{parent_key}[{index}]")
        else:
            documents.append(Document(page_content=f"{parent_key}: {data}"))

    parse_json(data)
    return documents


def create_vectorstore(documents: List[Document], index_path: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = []
    for doc in documents:
        docs.extend(text_splitter.split_documents([doc]))

    # Создание индекса FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


def search_in_index(query: str, index_path: str):
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=10)

    print("Результаты поиска:")
    for result in results:
        print(result.page_content)

    return results


def extract_phone_number(text: str, region: str = "RU") -> str | None:
    """
    Извлекает телефонный номер из текста и нормализует его к формату E.164.
    :param text: Текст, в котором может быть номер.
    :param region: Регион для распознавания номера, по умолчанию "RU".
    :return: Первый найденный номер в формате E.164 или None.
    """
    for match in phonenumbers.PhoneNumberMatcher(text, region):
        if phonenumbers.is_valid_number(match.number):
            return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
    return None


def save_user_phone_request(query: str, phone_log_path: str):
    normalized_phone = extract_phone_number(query)

    if normalized_phone:
        description = generate_description(query)

        try:
            if os.path.exists(phone_log_path):
                with open(phone_log_path, "r", encoding="utf-8") as file:
                    phone_logs = json.load(file)
            else:
                phone_logs = []
        except FileNotFoundError:
            phone_logs = []

        phone_logs.append({"phone": normalized_phone, "description": description})

        with open(phone_log_path, "w", encoding="utf-8") as file:
            json.dump(phone_logs, file, ensure_ascii=False, indent=4)

        return f"Ваш номер телефона {normalized_phone} сохранён. Менеджер свяжется с вами."
    else:
        return None


def generate_description(query: str) -> str:
    """
    Генерирует краткое описание запроса пользователя.
    :param query: Полный текст запроса.
    :return: Краткое описание.
    """
    prompt = (
        f"Пользователь задал следующий вопрос или сделал запрос: '{query}'.\n"
        "Сформируй краткое и понятное описание, чтобы менеджер понял суть запроса. "
        "Например, если пользователь хочет купить ноутбук 1999 года, опиши это как 'Пользователь интересуется покупкой ноутбука выпуска 1999 года'."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    description = llm.predict(prompt)
    return description.strip()


def generate_response(query: str, index_path: str):
    results = search_in_index(query, index_path)

    # Отбираем релевантные результаты
    if results:
        relevant_data = "\n".join([result.page_content for result in results])
        context = f"Вот что удалось найти:\n{relevant_data}" if relevant_data else "Данные отсутствуют."
    else:
        context = "Данные отсутствуют."

    messages = st.session_state["chat_history"]
    messages.append({"role": "user", "content": query})

    prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"{context}\n\n"
        "Используя предоставленные данные, постарайся дать точный ответ на вопрос пользователя. "
        "Если данных недостаточно или ответ не может быть однозначным, предложи пользователю оставить свой номер телефона "
        "для связи с менеджером."
    )

    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    response = llm.predict(prompt)

    messages.append({"role": "assistant", "content": response})

    return response


@st.cache_resource
def initialize_data():
    documents = load_json_as_documents(JSON_PATH)
    if not os.path.exists(f"{INDEX_PATH}.faiss"):
        create_vectorstore(documents, INDEX_PATH)


if __name__ == "__main__":
    initialize_data()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "system", "content": "Ты умный ассистент."}]

    # Интерфейс Streamlit
    st.title("Умный консультант с контекстом")
    st.write("Задавайте вопросы о товарах или просто общайтесь!")

    user_input = st.text_input("Ваш вопрос:", placeholder="Введите сообщение здесь...")

    if st.button("Отправить") and user_input.strip():
        phone_response = save_user_phone_request(user_input, PHONE_LOG_PATH)
        if phone_response:
            st.write(f"**Бот:** {phone_response}")
        else:
            with st.spinner("Генерирую ответ..."):
                response = generate_response(user_input, INDEX_PATH)
                st.write(f"**Бот:** {response}")

    st.subheader("История чата")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**Вы:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**Бот:** {msg['content']}")
