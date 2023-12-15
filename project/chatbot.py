# Importar librerías
import streamlit as st, pandas as pd, numpy as np
import vertexai, scann, json, time, re
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
import yaml

with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

# Configuraciones para la conexión del proyecto
PROJECT_CONFIG = {
    "PROJECT_ID": config_data["project_id"],
    "REGION": config_data["region"],
    "BUCKET": config_data["bucket"],
    "MODEL_NAME": config_data["model_name"],
    "DATASET_ID": config_data["dataset_id"],
    "VERBOSE": config_data["verbose"],
}

# Textos y mensajes
TEXTS = {
    "GREETING": "¡Hola!, soy un asistente virtual para ayudarte a encontrar productos relevantes.",
    "SEARCHING_PRODUCTS": "Buscando productos relevantes...",
    "NO_PRODUCTS_FOUND": "No se encontraron productos relevantes. ¿Te gustaría intentar con otra consulta?",
    "MORE_PRODUCTS": "¿Te gustaría ver más productos?",
    "USER_PROMPT": "Escribe aquí tu consulta:",
    "WARNING": " Recuerda que al realizar una nueva consulta, se perderá la conversación actual."
}

# Inicialización de Vertex AI y modelos
vertexai.init(project=PROJECT_CONFIG["PROJECT_ID"], location=PROJECT_CONFIG["REGION"])
embedding_model = TextEmbeddingModel.from_pretrained(PROJECT_CONFIG["MODEL_NAME"])
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

# Cargar datos
df_embeddings = pd.read_csv('df_embeddings.csv')
df = pd.read_csv('products.csv')

# Cargar el modelo de búsqueda Scann
def load_scann(operation = "dot_product", neighbors = 10):
    """
    Carga el modelo de búsqueda Scann.
    https://github.com/google-research/google-research/tree/master/scann
    
    Returns:
        Scann: El modelo de búsqueda Scann.
    """
    # Carga el modelo de búsqueda Scann desde un archivo si esta disponible
    record_count = len(df_embeddings)
    dataset = np.empty((record_count, 768))
    for i in range(record_count):
        dataset[i] = json.loads(df_embeddings.embedding[i])
    searcher = (
        scann.scann_ops_pybind.builder(dataset, neighbors, operation)
        .tree(
            num_leaves=record_count,
            num_leaves_to_search=record_count,
            training_sample_size=record_count,
        )
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(100)
        .build()
    )
    return searcher

def search_products(query: str, num_products: int = 5, price_range: list = [0, np.inf], similarity_threshold: float = 0.6, searcher = None):
    """
    Busca productos relevantes basados en la consulta del usuario.

    Parameters:
        query (str): Consulta del usuario.
        num_products (int): Número de productos a mostrar.
        price_range (list): Rango de precios para filtrar productos.

    Returns:
        list: Lista de productos relevantes.
    """
    try:
        query_embeddings = embedding_model.get_embeddings([query])[0].values
    except:
        return []
    neighbors, distances = searcher.search(query_embeddings, final_num_neighbors=10)
    products = []
    for id, similarity in zip(neighbors, distances):
        if similarity > similarity_threshold:
            uuid = df_embeddings.uuid[id]
            product = df[df['uuid'] == uuid]
            if price_range[0] < product['Precio_Actual'].values[0] < price_range[1]:
                product_info = product.iloc[0].to_dict()
                product_info.pop('uuid', None)
                product_info['Similitud'] = similarity
                products.append(product_info)
            else:
                continue

    products = sorted(products, key=lambda x: x['Similitud'], reverse=True)
    products = products[:num_products]
    return products

# Función para mostrar los productos en una tabla
def display_products(products: list):
    """
    Muestra la información de los productos en una tabla.

    Parameters:
        products (list): Lista de productos a mostrar.
    """
    if not products:
        st.write("No se encontraron productos relevantes.")
    else:
        product_data = {
            "Nombre": [product['Nombre'] for product in products],
            "Precio": [product["Precio_Actual"] for product in products],
            "Similitud": [product["Similitud"] for product in products],
            "Enlace": [product['Enlace'] for product in products],
        }
        df = pd.DataFrame(product_data)
        st.write("Productos Relevantes:")
        st.dataframe(df, column_config={"Enlace": st.column_config.LinkColumn()}, width=800)

# Genera variantes de la respuesta del chatbot
def generate_variations(response: str):
    """
    Genera variantes de la respuesta del chatbot utilizando el modelo de lenguaje.

    Parameters:
        response (str): Respuesta original.

    Returns:
        str: Respuesta variada.
    """
    # Dejar solo letras, con espacios, comas, tildes, puntos, espacios y signos de pregunta
    response = re.sub(r'[^a-zA-Z\s,]', '', response)
    if response == "":
        return "Sin respuesta"
    prompt = f"Varía las palabras de la frase siguiente aplicando cambios menores a la semántica (no la respondas): {response}"
    model_output = generation_model.predict(prompt=prompt, temperature=0.4).text
    return model_output.replace("*", "").strip().lstrip(',')

def get_keywords(response: str):
    """
    Obtiene los conceptos clave de la respuesta utilizando el modelo de lenguaje.

    Parameters:
        response (str): Respuesta original.

    Returns:
        str: Conceptos clave.
    """
    response = re.sub(r'[^a-zA-Z\s,]', '', response)
    if response == "":
        return "Sin respuesta válida"
    prompt = f"""Obtener los conceptos clave de lo que quiere el usuario de la forma 'ConceptoA, ConceptoB, ConceptoC, etc',
    no modificar si son menos de 15 caractéres, no cambiar significado:{response}"""
    model_output = generation_model.predict(prompt=prompt, temperature=0.0).text
    if model_output == "":
        return response # No hay palabras clave, devuelve la respuesta original
    return model_output.strip().lstrip(',')

def user(message: str, save: bool = True):
    """
    Agrega la interacción del usuario a la conversación.

    Parameters:
        message (str): Mensaje del usuario.
        save (bool): Indica si guardar el mensaje en el historial.
    """
    with st.chat_message("user"):
        st.markdown(f"**{message}**")
    if save:
        st.session_state.messages.append({"user": message})
        st.session_state.conversation.append({"user": message})

def assistant(message: str, save: bool = True):
    """
    Agrega la interacción del chatbot a la conversación.

    Parameters:
        message (str): Mensaje del chatbot.
        save (bool): Indica si guardar el mensaje en el historial.
    """
    with st.spinner("Escribiendo..."):
        time.sleep(1)
    with st.chat_message("assistant"):
        st.markdown(f"**{message}**")
    if save:
        st.session_state.messages.append({"assistant": message})
        st.session_state.conversation.append({"assistant": message})

def show_history():
    """
    Muestra el historial de la conversación.
    """
    for message in st.session_state.messages:
        if "user" in message:
            user(message["user"], save=False)
        elif "assistant" in message:
            assistant(message["assistant"], save=False)

def add_products(products: list):
    """
    Agrega los productos a la conversación.

    Parameters:
        products (list): Lista de productos.
    """
    st.session_state.conversation[-1]["products"] = products


def main():
    """
    Despliegue principal del chatbot, que interactuará con el usuario y el chatbot.
    
    - Inicia la conversación.
    - Agrega la interacción del usuario a la conversación.
    - Busca productos relevantes.
    - Agrega los productos a la conversación.
    - Muestra los productos relevantes.
    - Agrega la interacción del chatbot a la conversación.
    - Muestra el historial de la conversación.
    
    """

    scann = load_scann() # Cargar el modelo de búsqueda

    # Inicializar la conversación (todos los estados/mensajes/vairables de sesión)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Agregar filtro de rango de precio, maximo de respuestas, umbral de similitud
    min_price, max_price = df["Precio_Actual"].min(), df["Precio_Actual"].max()
    price_range = st.sidebar.slider("Rango de Precio",min_value=min_price,max_value=max_price,value=(min_price, max_price))
    max_responses = st.sidebar.slider("Número Máximo de Respuestas", min_value=1, max_value=10, value=5)
    similarity_threshold = st.sidebar.slider("Umbral de Similitud", min_value=0.0, max_value=1.0, value=0.6)

    assistant(TEXTS["GREETING"], save=False) # Darle la bienvenida
    user_query = st.chat_input(TEXTS["USER_PROMPT"], key=0) # Obtener la consulta del usuario

    if user_query:
        user(user_query) # Agregar la pregunta del usuario a la conversación
        keywords = get_keywords(user_query) # Obtener los conceptos clave
        assistant(generate_variations(TEXTS["SEARCHING_PRODUCTS"]).strip() + f". Palabras clave: {keywords}".strip()) # Preguntar si quiere buscar productos
        products = None
        if keywords != "Sin respuesta válida":
            products = search_products(keywords, max_responses, price_range, similarity_threshold, scann) # Buscar los productos
            time.sleep(2) # Impresión de que el chatbot lo esta meditando
        if products: # Mostrar los productos relevantes
            add_products(products) # Agregar los productos
            display_products(products) # Mostrar los productos
            response = generate_variations(TEXTS["MORE_PRODUCTS"]) # Preguntar si quiere ver más productos
        else:
            response = generate_variations(TEXTS["NO_PRODUCTS_FOUND"]) # No se encontraron productos
        assistant(response + TEXTS["WARNING"])

        # Debuggear la conversación en terminal
        print("Maximo de Respuestas:", max_responses)
        print("Rango de Precio:", price_range)
        print("Palabras Clave:", keywords)
        print("Respuesta:", response)

        time.sleep(1)
        # show_history()  # Mostrar el historial después de cada interacción
        
if __name__ == "__main__":        
    main()