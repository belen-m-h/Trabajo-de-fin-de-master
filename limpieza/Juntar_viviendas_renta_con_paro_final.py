import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


def create_dataframe(path, delimiter, encoding):
    """
    Crea un DataFrame a partir de un archivo CSV.

    Parámetros:
    - path (str): Ruta del archivo CSV.
    - delimiter (str): Delimitador para separar las columnas.
    - encoding (str): Codificación del archivo.

    Devuelve:
    - DataFrame.
    """
    return pd.read_csv(path, delimiter=delimiter, encoding=encoding)


def map_localities(paro, viviendas):
    """
    Mapea localidades entre dos DataFrames usando la similitud de cadenas.

    Parámetros:
    - paro (DataFrame): DataFrame con datos de paro.
    - viviendas (DataFrame): DataFrame con datos de viviendas.

    Devuelve:
    - Diccionario con el mapeo de localidades.
    """
    localidades_paros = np.array(paro[' Municipio'].unique())
    localidades_viviendas = np.array(viviendas['NMUN'].unique())
    mapeo = {}

    for localidad1 in localidades_paros:
        for localidad2 in localidades_viviendas:
            similitud = fuzz.ratio(localidad1, localidad2)
            if similitud > 70:
                mapeo[localidad1] = localidad2
                break
        else:
            mapeo[localidad1] = None
    return mapeo


def process_and_merge_data(viviendas_path, paro_path, output_path=None):
    """
    Limpia y procesa los datos de viviendas y paro, los mapea y los combina.

    Parámetros:
    - viviendas_path (str): Ruta del archivo CSV de viviendas.
    - paro_path (str): Ruta del archivo CSV de paro.
    - output_path (str, opcional): Ruta del archivo CSV para guardar los datos combinados.

    Devuelve:
    - DataFrame con los datos combinados y procesados.
    """
    # Crear DataFrames para viviendas y paro
    viviendas = create_dataframe(viviendas_path, ',', 'latin1')
    paro = create_dataframe(paro_path, ',', 'latin1')

    # Mapeo de localidades
    mapeo = map_localities(paro, viviendas)

    # Añadir columna de localidades mapeadas a paro
    paro['localidad_mapeada'] = paro[' Municipio'].map(mapeo)

    # Añadir columnas de año y mes a viviendas
    viviendas['Fecha'] = pd.to_datetime(viviendas['Fecha'], dayfirst=True)
    viviendas['Year'] = viviendas['Fecha'].dt.year.astype(str)  # Asegurar que 'Year' sea string
    viviendas['Mes'] = viviendas['Fecha'].dt.month.astype(str)  # Asegurar que 'Mes' sea string

    paro['year'] = paro['year'].astype(str)  # Asegurar que 'year' sea string
    paro['mes'] = paro['mes'].astype(str)    # Asegurar que 'mes' sea string
    paro = paro.dropna(subset=['localidad_mapeada'])
    paro = paro.drop_duplicates(subset=['localidad_mapeada', 'year', 'mes'])

    # Unir los conjuntos de datos
    df_final = pd.merge(viviendas, paro, how='left', left_on=['NMUN', 'Year', 'Mes'], right_on=['localidad_mapeada', 'year', 'mes'])

    # Crear datasets separados para 2021 y 2022
    data21 = df_final[df_final['Year'] == '2021']
    data22 = df_final[df_final['Year'] == '2022']

    data21['Paro hombre edad < 25'] = data21['Paro hombre edad < 25'].astype('Int64')
    data21['Paro hombre edad < 25'] = data21['Paro hombre edad < 25'].fillna(data21['Paro hombre edad < 25'].mean().astype(int))

    data21['Paro hombre edad 25 -45 '] = data21['Paro hombre edad 25 -45 '].fillna(data21['Paro hombre edad 25 -45 '].mean().astype(int))

    data21['Paro hombre edad >=45'] = data21['Paro hombre edad >=45'].astype('Int64')
    data21['Paro hombre edad >=45'] = data21['Paro hombre edad >=45'].fillna(data21['Paro hombre edad >=45'].mean().astype(int))

    # data21['Renta bruta media por persona'] = data21['Renta bruta media por persona'].astype('Int64')
    data21['Renta bruta media por persona'] = data21['Renta bruta media por persona'].fillna(data21['Renta bruta media por persona'].mean().astype(int))

    # data21['Renta neta media por hogar'] = data21['Renta neta media por hogar'].astype('Int64')
    data21['Renta neta media por hogar'] = data21['Renta neta media por hogar'].fillna(data21['Renta neta media por hogar'].mean().astype(int))

    # data21['Renta neta media por persona '] = data21['Renta neta media por persona '].astype('Int64')
    data21['Renta neta media por persona '] = data21['Renta neta media por persona '].fillna(data21['Renta neta media por persona '].mean().astype(int))

    data21['total Paro Registrado'] = data21['total Paro Registrado'].astype('Int64')
    data21['total Paro Registrado'] = data21['total Paro Registrado'].fillna(data21['total Paro Registrado'].mean().astype(int))

    data21['Paro mujer edad < 25'] = data21['Paro mujer edad < 25'].astype('Int64')
    data21['Paro mujer edad < 25'] = data21['Paro mujer edad < 25'].fillna(data21['Paro mujer edad < 25'].mean().astype(int))

    data21['Paro mujer edad 25 -45 '] = data21['Paro mujer edad 25 -45 '].astype('Int64')
    data21['Paro mujer edad 25 -45 '] = data21['Paro mujer edad 25 -45 '].fillna(data21['Paro mujer edad 25 -45 '].mean().astype(int))

    data21['Paro mujer edad >=45'] = data21['Paro mujer edad >=45'].astype('Int64')
    data21['Paro mujer edad >=45'] = data21['Paro mujer edad >=45'].fillna(data21['Paro mujer edad >=45'].mean().astype(int))

    data21['Paro Agricultura'] = data21['Paro Agricultura'].astype('Int64')
    data21['Paro Agricultura'] = data21['Paro Agricultura'].fillna(data21['Paro Agricultura'].mean().astype(int))

    data21['Paro Industria'] = data21['Paro Industria'].astype('Int64')
    data21['Paro Industria'] = data21['Paro Industria'].fillna(data21['Paro Industria'].mean().astype(int))

    data21['Paro construccion'] = data21['Paro construccion'].astype('Int64')
    data21['Paro construccion'] = data21['Paro construccion'].fillna(data21['Paro construccion'].mean().astype(int))

    data21['Paro Servicios'] = data21['Paro Servicios'].astype('Int64')
    data21['Paro Servicios'] = data21['Paro Servicios'].fillna(data21['Paro Servicios'].mean().astype(int))

    data21['Paro Sin empleo Anterior'] = data21['Paro Sin empleo Anterior'].astype('Int64')
    data21['Paro Sin empleo Anterior'] = data21['Paro Sin empleo Anterior'].fillna(data21['Paro Sin empleo Anterior'].mean().astype(int))
    # Imputar datos faltantes con la media para cada año
    data22['Paro hombre edad < 25'] = data22['Paro hombre edad < 25'].apply(to_numeric_or_nan)

    data22['Paro hombre edad < 25'] = data22['Paro hombre edad < 25'].astype('Int64')
    data22['Paro hombre edad < 25'] = data22['Paro hombre edad < 25'].fillna(data22['Paro hombre edad < 25'].mean().astype(int))

    data22['Paro hombre edad 25 -45 '] = data22['Paro hombre edad 25 -45 '].apply(to_numeric_or_nan)
    data22['Paro hombre edad 25 -45 '] = data22['Paro hombre edad 25 -45 '].fillna(data22['Paro hombre edad 25 -45 '].mean().astype(int))

    data22['Paro hombre edad >=45'] = data22['Paro hombre edad >=45'].apply(to_numeric_or_nan)
    data22['Paro hombre edad >=45'] = data22['Paro hombre edad >=45'].fillna(data22['Paro hombre edad >=45'].mean().astype(int))

    # data22['Renta bruta media por persona'] = data22['Renta bruta media por persona'].astype('Int64')
    data22['Renta bruta media por persona'] = data22['Renta bruta media por persona'].fillna(data22['Renta bruta media por persona'].mean().astype(int))

    # data22['Renta neta media por hogar'] = data22['Renta neta media por hogar'].astype('Int64')
    data22['Renta neta media por hogar'] = data22['Renta neta media por hogar'].fillna(data22['Renta neta media por hogar'].mean().astype(int))

    # data22['Renta neta media por persona '] = data22['Renta neta media por persona '].astype('Int64')
    data22['Renta neta media por persona '] = data22['Renta neta media por persona '].fillna(data22['Renta neta media por persona '].mean().astype(int))

    data22['total Paro Registrado'] = data22['total Paro Registrado'].apply(to_numeric_or_nan)
    data22['total Paro Registrado'] = data22['total Paro Registrado'].fillna(data22['total Paro Registrado'].mean().astype(int))

    data22['Paro mujer edad < 25'] = data22['Paro mujer edad < 25'].apply(to_numeric_or_nan)
    data22['Paro mujer edad < 25'] = data22['Paro mujer edad < 25'].fillna(data22['Paro mujer edad < 25'].mean().astype(int))

    data22['Paro mujer edad 25 -45 '] = data22['Paro mujer edad 25 -45 '].apply(to_numeric_or_nan)
    data22['Paro mujer edad 25 -45 '] = data22['Paro mujer edad 25 -45 '].fillna(data22['Paro mujer edad 25 -45 '].mean().astype(int))

    data22['Paro mujer edad >=45'] = data22['Paro mujer edad >=45'].apply(to_numeric_or_nan)
    data22['Paro mujer edad >=45'] = data22['Paro mujer edad >=45'].fillna(data22['Paro mujer edad >=45'].mean().astype(int))

    data22['Paro Agricultura'] = data22['Paro Agricultura'].apply(to_numeric_or_nan)
    data22['Paro Agricultura'] = data22['Paro Agricultura'].fillna(data22['Paro Agricultura'].mean().astype(int))

    data22['Paro Industria'] = data22['Paro Industria'].apply(to_numeric_or_nan)
    data22['Paro Industria'] = data22['Paro Industria'].fillna(data22['Paro Industria'].mean().astype(int))

    data22['Paro construccion'] = data22['Paro construccion'].apply(to_numeric_or_nan)
    data22['Paro construccion'] = data22['Paro construccion'].fillna(data22['Paro construccion'].mean().astype(int))

    data22['Paro Servicios'] = data22['Paro Servicios'].apply(to_numeric_or_nan)
    data22['Paro Servicios'] = data22['Paro Servicios'].fillna(data22['Paro Servicios'].mean().astype(int))

    data22['Paro Sin empleo Anterior'] = data22['Paro Sin empleo Anterior'].apply(to_numeric_or_nan)
    data22['Paro Sin empleo Anterior'] = data22['Paro Sin empleo Anterior'].fillna(data22['Paro Sin empleo Anterior'].mean().astype(int))


    # Eliminar columnas innecesarias
    data21 = data21.drop([' Municipio', 'year', 'mes', 'localidad_mapeada'], axis=1)
    data22 = data22.drop([' Municipio', 'year', 'mes', 'localidad_mapeada'], axis=1)

    # Concatenar datasets de 2021 y 2022
    df_final = pd.concat([data21, data22], ignore_index=True)

    # Guardar DataFrame limpios2 (opcional)
    if output_path:
        df_final.to_csv(output_path, index=False)

    return df_final


def to_numeric_or_nan(value):
    """
    Convierte un valor a numérico o devuelve NaN si no es posible.

    Parámetros:
    - value: Valor a convertir.

    Devuelve:
    - Valor numérico o NaN.
    """
    try:
        return pd.to_numeric(value)
    except ValueError:
        return np.nan

def check_missing_data(df):
    # 1. Comprobar si hay algún NaN en todo el DataFrame
    print(f"¿Hay algún NaN en el DataFrame? {df.isnull().values.any()}")

    # 2. Comprobar la cantidad de NaN por columna

    print("Cantidad de NaN por columna:\n", df.isnull().sum())

    # 3. Mostrar las filas que contienen NaN
    filas_con_nan = df[df.isnull().any(axis=1)]
    print("Filas que contienen NaN:\n", filas_con_nan)



if __name__ == '__main__':
    viviendas_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosViviendasRentaMadrid.csv'
    paro_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosParoMadrid.csv'
    output_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosCompletos.csv'

    cleaned_data = process_and_merge_data(viviendas_path, paro_path, output_path=output_path)
    check_missing_data(cleaned_data)
    print(cleaned_data.shape)
    print(cleaned_data.describe())
