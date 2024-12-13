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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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


def save_user_phone_request(user_input: str, history: list, phone_log_path: str):
    normalized_phone = extract_phone_number(user_input)

    if normalized_phone:
        description = generate_description(history)

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

        return f"Спасибо. Менеджер свяжется с вами в ближайшее время."
    else:
        return None


def generate_description(history: list) -> str:
    """
    Генерирует краткое описание на основе истории ввода пользователя.
    :param history: История сообщений пользователя (список словарей с ключами 'role' и 'content').
    :return: Краткое описание запроса пользователя.
    """
    # Преобразуем историю в читаемый формат
    user_history = "\n".join(
        f"Пользователь: {msg['content']}" for msg in history if msg["role"] == "user"
    )

    # Промпт для генерации описания
    prompt = (
        f"Вот история сообщений пользователя:\n{user_history}\n\n"
        "На основе этой истории сформируй краткое и понятное описание запроса пользователя. "
        "Старайся быть максимально точным и кратким. Например:\n"
        "- Если пользователь интересуется покупкой ноутбука, опиши это как 'Пользователь интересуется покупкой "
        "ноутбука'.\n"
        "- Если пользователь хочет заказать доставку, опиши это как 'Пользователь хочет заказать доставку'.\n"
        "- Если пользователь спрашивает о количестве товара, опиши это как 'Пользователь интересуется количеством "
        "товара'.\n"
        "Сформулируй одно предложение, которое описывает суть запроса пользователя."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    description = llm.predict(prompt)
    return description.strip()


def determine_object(response: str, relevant_data: str) -> str:
    """
    Определяет предмет, о котором идёт речь, на основании ответа и данных.
    :param response: Ответ GPT.
    :param relevant_data: Данные, найденные в JSON.
    :return: Название предмета.
    """
    prompt = (
        f"Ответ, данный пользователю: '{response}'.\n\n"
        f"Вот что удалось найти в данных. result_search:\n{relevant_data}\n\n"
        "На основании ответа и данных определи, о каком конкретном предмете идёт речь. "
        "Верни всю доступную информацию о предмете. Без лишней воды, например название, количество, цену и т.д. "
        "Если предмет невозможно определить, верни 'Неизвестно'."
        "Ответ должен быть структурированным и понятным и содержать конкретные данные из result_search."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    determined_object = llm.predict(prompt)
    return determined_object.strip()


def group_and_structure_data(raw_data: str) -> str:
    """
    Отправляет данные в LLM для группировки и структурирования.

    :param raw_data: Строка с сырыми данными для обработки.
    :return: Структурированный результат в удобочитаемом виде.
    """
    prompt = (
        f"У тебя есть следующие данные:\n\n"
        f"{raw_data}\n\n"
        "Сгруппируй эти данные по логическим категориям или типам объектов. Укажи только категории и сгруппированные "
        "элементы, без дублирующихся данных. "
        "Представь результат в структурированном и удобочитаемом виде."
    )

    # Отправляем запрос в OpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    structured_result = llm.predict(prompt)

    return structured_result.strip()


def convert_to_messages(history):
    """
    Конвертирует историю чата из словарей в объекты LangChain.
    :param history: Список словарей с ключами "role" и "content".
    :return: Список объектов HumanMessage, AIMessage или SystemMessage.
    """
    messages = []
    for message in history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            raise ValueError(f"Неизвестная роль: {role}")
    return messages


def generate_response(query: str, index_path: str):
    # Выполняем поиск в индексе
    results = search_in_index(query, index_path)

    # Формируем контекст из результатов поиска
    relevant_data = None
    if results:
        relevant_data = "\n".join([result.page_content for result in results])
        context = f"Вот что удалось найти:\n{relevant_data}" if relevant_data else "Данные отсутствуют."
    else:
        context = "Данные отсутствуют."

    # Добавляем текущий предмет разговора в контекст, если он есть
    current_item = st.session_state.get("current_item", None)
    if current_item:
        context = f"Текущий предмет разговора: {current_item}.\n{context}"

    # Добавляем текущий запрос пользователя в историю
    st.session_state["chat_history"].append({"role": "user", "content": query})

    # Конвертируем историю чата в формат LangChain
    messages = convert_to_messages(st.session_state["chat_history"])

    # Формируем сообщение с контекстом
    if context and context != "Данные отсутствуют.":
        messages.append(SystemMessage(content=f"Контекст: {context}"))

    # Запрашиваем GPT через LangChain
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    response_message = llm.predict_messages(messages)

    # Добавляем ответ ассистента в историю
    st.session_state["chat_history"].append({"role": "assistant", "content": response_message.content})

    # Обновляем текущий предмет разговора
    if relevant_data:
        grouped_data = group_and_structure_data(relevant_data)
        current_item = determine_object(response_message.content, grouped_data)
        st.session_state["current_item"] = current_item

    return response_message.content



@st.cache_resource
def initialize_data():
    documents = load_json_as_documents(JSON_PATH)
    if not os.path.exists(f"{INDEX_PATH}.faiss"):
        create_vectorstore(documents, INDEX_PATH)


if __name__ == "__main__":
    initialize_data()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "system",
                "content": (
                    "Ты умный ассистент. "
                    "Не используй словосочетания типа 'согласно предоставленным данным' или 'в списке'. "
                    "Ответы должны быть максимально простыми, естественными и без лишних формальностей. И чтобы "
                    "пользователь получил максимум конкретной информации"
                    "Если ты не смог найти информацию то предложе пользователю оставить номер чтобы с ним связался "
                    "менеджер."
                )
            }
        ]

    # Интерфейс Streamlit
    st.title("Умный консультант с контекстом")
    st.write("Задавайте вопросы о товарах или просто общайтесь!")

    user_input = st.text_input("Ваш вопрос:", placeholder="Введите сообщение здесь...")

    if st.button("Отправить") and user_input.strip():
        phone_response = save_user_phone_request(user_input, st.session_state["chat_history"], PHONE_LOG_PATH)
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
