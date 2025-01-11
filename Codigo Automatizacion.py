# Código Python para el sistema automatizado de análisis de datos


import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
from tqdm import tqdm
from deep_translator import GoogleTranslator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import smogn
import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -------------------------------------------------------------------
# CONFIGURACIÓN DE GSPREAD
# -------------------------------------------------------------------
# 2. Carga tus credenciales desde un archivo JSON (ajusta la ruta al tuyo).
#    Este archivo debe contener la clave de servicio (Service Account Key).
#creds = ServiceAccountCredentials.from_json_keyfile_name("jjson.json", scope)

# Leer credenciales desde la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
json_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not json_creds:
    raise ValueError("Las credenciales de Google no están configuradas en los Secrets de GitHub como GOOGLE_APPLICATION_CREDENTIALS.")
# Parsear el JSON de credenciales
creds_dict = json.loads(json_creds)
# Crear credenciales de servicio
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
# Autorizar gspread
gc = gspread.authorize(creds)


# -------------------------------------------------------------------
# FUNCIÓN PARA NORMALIZAR RUT
# -------------------------------------------------------------------
def normalizar_rut(rut_str):
    if not isinstance(rut_str, str) or not rut_str.strip():
        return None
    rut_str = rut_str.upper().replace(".", "").replace("-", "").replace(" ", "")
    cuerpo = rut_str[:-1]
    dv = rut_str[-1]
    if not cuerpo.isdigit():
        return None
    cuerpo_invertido = cuerpo[::-1]
    trozos = [cuerpo_invertido[i:i+3] for i in range(0, len(cuerpo_invertido), 3)]
    cuerpo_formateado = ".".join(trozos)[::-1]
    rut_normalizado = f"{cuerpo_formateado}-{dv}"
    return rut_normalizado

# -------------------------------------------------------------------
# MODELOS DE SENTIMIENTOS Y EMOCIONES
# -------------------------------------------------------------------
# Sentimientos (nlptown/bert-base-multilingual-uncased-sentiment)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=512,
)

# Emociones en inglés (j-hartmann/emotion-english-distilroberta-base)
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Lista de emociones manejadas por el modelo
emociones_ingles = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# -------------------------------------------------------------------
# FUNCIONES DE AYUDA
# -------------------------------------------------------------------
def analizar_sentimientos(columna_serie):
    """
    Aplica análisis de sentimientos a la Serie recibida.
    Retorna un valor entero de 1..5 según la cantidad de 'stars' detectada en la etiqueta del modelo.
    """
    def convertir_a_numero(texto):
        if not isinstance(texto, str) or not texto.strip():
            return None
        resultado = sentiment_analyzer(texto)[0]
        label = resultado.get("label", "").lower()
        match = re.match(r'(\d)\s*star', label)
        if match:
            return int(match.group(1))
        return None

    return columna_serie.apply(convertir_a_numero)

translator = GoogleTranslator(source='es', target='en')

def obtener_emociones_con_traduccion(texto):
    if not isinstance(texto, str):
        return {}
    texto = texto.strip()
    if len(texto) < 2:
        return {}
    try:
        texto_en = translator.translate(text=texto)
        resultados = emotion_analyzer(texto_en)
        if isinstance(resultados, list) and len(resultados) > 0 and isinstance(resultados[0], list):
            return {item["label"]: item["score"] for item in resultados[0]}
    except Exception as e:
        print(f"Error procesando texto: '{texto}', Error: {e}")
    return {}

def extraer_prefijo(col_name):
    """
    Dada una columna como '4.1 ¿Cuál..._anger', retorna ('4.1', 'anger').
    Si fuera 'Texto_Combinado_anger', retorna ('Texto_Combinado', 'anger').
    """
    partes = col_name.rsplit('_', 1)
    if len(partes) == 2:
        posible_pregunta, emocion = partes
        match = re.match(r'^(4\.\d+)', posible_pregunta.strip())  # p.ej. '4.1'
        if match:
            return match.group(1), emocion
        if 'Texto_Combinado' in posible_pregunta:
            return 'Texto_Combinado', emocion
        return posible_pregunta, emocion
    return col_name, None

def renombrar_col_emocion(col_name):
    """
    Convierte algo como '4.1 ¿Cuál..._anger' en '4.1_emocion_anger'
    o 'Texto_Combinado_anger' en 'Texto_Combinado_emocion_anger'.
    """
    prefix, suffix = extraer_prefijo(col_name)
    if suffix:
        return f"{prefix}_emocion_{suffix}"
    return col_name

def extraer_cat123(df_in):
    """
    Devuelve un DataFrame con las columnas que empiezan con 1.X, 2.X o 3.X
    y las renombra a su formato corto (ej. '1.1 Pregunta...' => '1.1').
    """
    rename_map = {}
    for col in df_in.columns:
        match = re.match(r'^([1-3]\.\d+)', col.strip())
        if match:
            short_name = match.group(1)
            rename_map[col] = short_name
    df_cat123 = df_in[list(rename_map.keys())].rename(columns=rename_map)
    return df_cat123

# 2) Convertimos Timestamps a str
def convertir_timestamp_a_str(valor):
    if isinstance(valor, pd.Timestamp):
        # Ajusta el formato a tu gusto
        return valor.strftime("%Y-%m-%d %H:%M:%S")
    return valor

# -------------------------------------------------------------------
# LISTA DE [CARRERA, SPREADSHEET_ID] PARA RECORRER EN EL FOR
# -------------------------------------------------------------------
lista_carreras_spreadsheets = [
    # Facultad: Ingeniería, Ciencia y Tecnología
    ["Ingeniería, Ciencia y Tecnología", "Ingeniería en Realidad Virtual y Diseño de Videojuegos Digitales", "1hvmrFY2mWxwydQCXrJuuqGTJzQiaRe8smyQjye58ZGs"],
    ["Ingeniería, Ciencia y Tecnología", "Contador Auditor", "1k8qX772KtWFzXN-7pmoQarDPSOxZHiSq9Dxb9y5ovN8"],
    ["Ingeniería, Ciencia y Tecnología", "Ingeniería en Informática", "1p24iQ1dWLhRd6UYn3dMmputf9XRwIpsQ2M3nWBVx5FI"],

    # Facultad: Ciencias Humanas
    ["Ciencias Humanas", "Derecho", "1rcJkRDmI4zwAo4AZ0_sitLbFg_VpU_-e0xLYlVsPt_8"],
    ["Ciencias Humanas", "Periodismo", "13rPbX2ZvtWcAinluVm_vmPoSLS4Nk00org5YDj9H8S8"],
    ["Ciencias Humanas", "Trabajo Social", "1nFq7NRmCxXwZae4GGmU_5IajF6D0EfhDIT9FxpMYqQc"],
    ["Ciencias Humanas", "Psicología", "1m9nf9ckOkXIHWn6Vm_agT9_bGXXE-JzroM6uybLIYu8"],
    ["Ciencias Humanas", "Pedagogía para Educación General Básica", "1UaxcKpWK-Pat-zwPrCWY6Xh0zQ0fCjEP4RoTI1_uV9I"],
    ["Ciencias Humanas", "Pedagogía en Educación Parvularia", "1Gx_Hddt1f32zonPqJ21UWclTp9G2kILFqsaAbrSl7mY"],
    ["Ciencias Humanas", "Pedagogía en Educación Física, Deporte y Recreación", "1PW4NSKpPbfylVL2r_Fls83IZNeFWzitEePuJVn8GQxw"],

    # Facultad: Ciencias de la Salud
    ["Ciencias de la Salud", "Kinesiología", "1BGB5mlUmQ8DPaMWzAwNLk-zZJ6AbXBNvaddWMeYhB2o"],
    ["Ciencias de la Salud", "Nutrición y Dietética", "1ZnyZZ2pNvgbKHZMoXRFXmmXwMrmagbZCrxE7uRb_bts"],
    ["Ciencias de la Salud", "Tecnología Médica IMAGENOLOGÍA", "12cbyMoYItjrhQ4nNzt07taXtwr3umCRh6gOjCB0o7p0"],
    ["Ciencias de la Salud", "Tecnología Médica OFTALMOLOGÍA", "1_8qCw3NB3ELxE30wGeJeTn4lEbl95vXs3KxBnyIqsCE"],
    ["Ciencias de la Salud", "Fonoaudiología", "1rZn_7OYlp_b2YzjFiEn3-MaABOXpVBiQK3TLfVrxbJE"],

    # Facultad: Ciencias Médicas
    ["Ciencias Médicas", "Medicina Veterinaria", "1pc00nez7AVXoMdHS9yF4JDQHosAh7jQfzZYkiuSA8xI"],
    ["Ciencias Médicas", "Enfermería", "19a3Clz4VT0V4qq-dU3sb9gJkm5We3-bymF12PM-nC_w"],
    ["Ciencias Médicas", "Obstetricia y Puericultura", "1vKRk3dOm5raedbpLyFwsJvacKs4FO3gXAg1Jn5s2bvE"]
]
#lista_carreras_spreadsheets = [
#    ["Ingeniería en Realidad Virtual y Diseño de Videojuegos Digitales", "1hvmrFY2mWxwydQCXrJuuqGTJzQiaRe8smyQjye58ZGs"],
#    ["Contador Auditor", "1k8qX772KtWFzXN-7pmoQarDPSOxZHiSq9Dxb9y5ovN8"],
#    ["Psicología", "1m9nf9ckOkXIHWn6Vm_agT9_bGXXE-JzroM6uybLIYu8"],
#]

# -------------------------------------------------------------------
# BUCLE PRINCIPAL
# -------------------------------------------------------------------
for facultad, carrera, spreadsheet_id in lista_carreras_spreadsheets:
    print(f"Procesando carrera: {carrera} | SpreadsheetID: {spreadsheet_id}")

    # 1) Abrimos el Spreadsheet y leemos la hoja "Respuestas de formulario 1"
    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.worksheet("Respuestas de formulario 1")
    data = worksheet.get_all_records()

    # Lo convertimos en DataFrame
    df = pd.DataFrame(data)

    # Si tu columna de timestamp se llama distinto, ajusta el nombre abajo:
    if 'Marca temporal' in df.columns:
        df['Marca temporal'] = pd.to_datetime(df['Marca temporal'], dayfirst=True, errors='coerce')

    # 2) Normalizar nombres (columna C)
    nombre_col = 'Por favor, indique su nombre completo (NOMBRE APELLIDO 1 APELLIDO 2)'
    if nombre_col in df.columns:
        df[nombre_col] = df[nombre_col].str.lower().str.strip()

    # 3) Normalizar RUT (columna D)
    rut_col = 'Por favor, indique su RUT (XX.XXX.XXX-X)'
    if rut_col in df.columns:
        df[rut_col] = df[rut_col].apply(normalizar_rut)

    # 4) Eliminar duplicados de RUT, quedándote solo con la última respuesta
    if 'Marca temporal' in df.columns and rut_col in df.columns:
        df.sort_values(by='Marca temporal', ascending=False, inplace=True)
        df.drop_duplicates(subset=rut_col, keep='first', inplace=True)

    # ----------------------------------------------------------------
    # Detección de columnas 4.1 / 4.2 / etc. para análisis de texto
    # ----------------------------------------------------------------
    columnas_categoria_4 = [col for col in df.columns if col.strip().startswith('4.1') or col.strip().startswith('4.2')]
    categoria_4 = df[columnas_categoria_4].copy() if columnas_categoria_4 else pd.DataFrame()

    if not categoria_4.empty:
        categoria_4['Texto_Combinado'] = categoria_4.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        categoria_4 = categoria_4[categoria_4['Texto_Combinado'].str.strip() != '']

        # Análisis de sentimiento
        for columna in categoria_4.columns:
            if columna != 'Texto_Combinado':
                categoria_4[f'{columna}_Sentimiento'] = analizar_sentimientos(categoria_4[columna])

        #tqdm.pandas()
        columns_to_analyze = categoria_4.columns[:3]  # Ajusta si corresponde
        resultados_emociones = pd.DataFrame(index=categoria_4.index)

        for col in columns_to_analyze:
            dict_emociones_columna = categoria_4[col].apply(obtener_emociones_con_traduccion)
             #dict_emociones_columna = categoria_4[col].progress_apply(obtener_emociones_con_traduccion)
            for em in emociones_ingles:
                col_name = f"{col}_{em}"
                resultados_emociones[col_name] = dict_emociones_columna.apply(lambda x: x.get(em, 0.0))
                #resultados_emociones[col_name] = dict_emociones_columna.apply(lambda x: x.get(em, 0.0) if x else 0.0)
        # Renombrar columnas de emociones
        df_emociones_ren = resultados_emociones.copy()
        df_emociones_ren.columns = [renombrar_col_emocion(c) for c in df_emociones_ren.columns]
    else:
        categoria_4 = pd.DataFrame()
        df_emociones_ren = pd.DataFrame()

    # ----------------------------------------------------------------
    # Unir categorías 1..3 (si existen) y luego la parte de 4.x
    # ----------------------------------------------------------------
    df_cat123 = extraer_cat123(df)

    df_limpio = pd.DataFrame(index=df.index)
    df_limpio = df_limpio.join(df_cat123, how='left')

    if not categoria_4.empty:
        # Seleccionar columnas que contienen valores de sentimientos (evitar texto)
        cols_sentimientos = [col for col in categoria_4.columns if "_Sentimiento" in col]
        cols_texto_combinado = [col for col in categoria_4.columns if "Texto_Combinado" in col]

        # Verificar que existen las columnas de emociones (generadas previamente)
        cols_emociones = [col for col in df_emociones_ren.columns if "_emocion_" in col]

        # Combinar todas las columnas relevantes
        columnas_a_unir = cols_sentimientos + cols_texto_combinado

        # Unir columnas de emociones si existen
        if cols_emociones:
            df_emociones = df_emociones_ren[cols_emociones]
            df_limpio = df_limpio.join(df_emociones, how="left")

        # Unir columnas de sentimientos y texto combinado
        if columnas_a_unir:
            df_limpio = df_limpio.join(categoria_4[columnas_a_unir], how="left")

        # Renombrar las columnas de forma explícita
        rename_map = {}
        if cols_texto_combinado:
            rename_map[cols_texto_combinado[0]] = "Texto_Combinado_sentimientos"
        if len(cols_sentimientos) >= 2:
            rename_map.update({
                cols_sentimientos[0]: "4.1_sentimientos",
                cols_sentimientos[1]: "4.2_sentimientos",
            })

        # Aplicar renombrado
        df_limpio.rename(columns=rename_map, inplace=True)

    # ----------------------------------------------------------------
    # Extraer la parte desde la primera columna "4.1" hasta el final
    # ----------------------------------------------------------------
    idx_primera_4_1 = None
    for i, col_name in enumerate(df_limpio.columns):
        if col_name.strip().startswith("4.1"):
            idx_primera_4_1 = i
            break

    if idx_primera_4_1 is not None:
        columnas_desde_4_1  = df_limpio.columns[idx_primera_4_1:]
        df_limpio_filtrado  = df_limpio[columnas_desde_4_1]
        df_temporal         = pd.concat([df, df_limpio_filtrado], axis=1)
    else:
        df_temporal         = df.copy()  # Por si no se halló ninguna "4.1"

    # ----------------------------------------------------------------
    # GUARDAR df_temporal EN UNA NUEVA HOJA DEL MISMO SPREADSHEET
    # ----------------------------------------------------------------
    # 1) Si ya existiera una hoja "df_temporal", podemos borrarla para sobreescribir
    try:
        hoja_existente = sh.worksheet("df_temporal")
        sh.del_worksheet(hoja_existente)
    except:
        pass  # Si no existe, no pasa nada

    # 2) Crear nueva hoja y volcar los datos
    nueva_hoja = sh.add_worksheet(title="df_temporal",
                                  rows=len(df_temporal)+10,
                                  cols=len(df_temporal.columns)+10)

    # 1) Creamos una copia de df_temporal
    df_temporal_str = df_temporal.copy()
    # 3) Aplicar la función a todo el DataFrame
    df_temporal_str = df_temporal_str.applymap(convertir_timestamp_a_str)

    # 4) Rellenar NaN con cadena vacía (opcional)
    df_temporal_str = df_temporal_str.fillna("")

    # gspread necesita listas anidadas para update: primera fila son encabezados,
    # luego cada fila de datos
    encabezados = df_temporal_str.columns.tolist()
    valores     = df_temporal_str.values.tolist()
    nueva_hoja.update([encabezados] + valores)


