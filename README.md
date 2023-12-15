# E-Commerce-Chatbot

Este proyecto es un chatbot desarrollado para interactuar con los clientes de Palacio de Hierro, una reconocida tienda departamental. El chatbot está diseñado para proporcionar información sobre productos y precios ante preguntas por recomendaciones de los clientes de manera eficiente y amigable.

## Ejecución del Web Scraping

Para obtener los datos necesarios para el chatbot, realizamos web scraping en el sitio web de Palacio de Hierro. Aquí están los pasos para ejecutar el script de web scraping:

1. Instalar las librerías necesarias para ejecutar el script de web scraping. Para esto, ejecutar el siguiente comando en la terminal:

```bash
pip install -r requirements.txt
```

1. Ejecutar los scripts en el orden que se indica a continuación:

- scrap_to_json.ipynb: Este script obtiene los links de los productos de la página principal de la tienda y los JSON de cada subcategoría de productos.
- json_to_csv.ipynb: Este script convierte los JSON obtenidos en el paso anterior a un dataframe de pandas en formato CSV.
- csv_to_test-csv.ipynb: Este script toma el CSV obtenido en el paso anterior y extrae una muestra de productos para probar el chatbot.
  
## Ejecución del Chatbot

Para ejecutar el chatbot, se debe cargar su base de datos de productos completa o de test en GCP. Puede hacerlo moviendo el archivo para embeddings y de productos creados a la carpeta project dentro de un notebook en Vertex AI Workbench o cargarlos directamente en un bucket de GCP mediante BigQuery.

Luego de esto, ejecute el notebook en Vertex AI Workbench llamado "get_embeddings.ipynb" y siga las instrucciones que se encuentran en él. Este notebook se encargará de crear los embeddings de los productos y guardarlos posteriormente.

Finalmente, ejecute la aplicación de Streamlit mediante el siguiente comando en la terminal:

```bash
streamlit run chatbot.py
```

## Autores
- Sebastián Guzmán
- Diana Escobar
- Valentina Castro