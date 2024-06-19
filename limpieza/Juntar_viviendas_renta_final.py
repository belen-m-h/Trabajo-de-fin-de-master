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


def map_localities(rentas, viviendas):
    """
    Mapea localidades entre dos DataFrames usando la similitud de cadenas.

    Parámetros:
    - rentas (DataFrame): DataFrame con datos de rentas.
    - viviendas (DataFrame): DataFrame con datos de viviendas.

    Devuelve:
    - Diccionario con el mapeo de localidades.
    """
    localidades_rentas = np.array(rentas['Municipios'].unique())
    localidades_viviendas = np.array(viviendas['NMUN'].unique())
    mapeo = {}

    for localidad1 in localidades_rentas:
        for localidad2 in localidades_viviendas:
            similitud = fuzz.ratio(localidad1, localidad2)
            if similitud > 70:
                mapeo[localidad1] = localidad2
                break
        else:
            mapeo[localidad1] = None
    return mapeo


def process_and_merge_data(viviendas_path, renta_path, output_path=None):
    """
    Limpia y procesa los datos de viviendas y rentas, los mapea y los combina.

    Parámetros:
    - viviendas_path (str): Ruta del archivo CSV de viviendas.
    - renta_path (str): Ruta del archivo CSV de rentas.
    - output_path (str, opcional): Ruta del archivo CSV para guardar los datos combinados.

    Devuelve:
    - DataFrame con los datos combinados y procesados.
    """
    # Crear DataFrames para viviendas y rentas
    viviendas = create_dataframe(viviendas_path, ',', 'latin1')
    renta = create_dataframe(renta_path, ',', 'latin1')

    # Mapeo de localidades
    mapeo = map_localities(renta, viviendas)

    # Añadir columna de localidades mapeadas a rentas
    renta['localidad_mapeada'] = renta['Municipios'].map(mapeo)
    renta = renta.dropna(subset=['localidad_mapeada'])

    # Añadir columnas de año y mes a viviendas
    viviendas['Fecha'] = pd.to_datetime(viviendas['Fecha'], dayfirst=True)
    viviendas['Year'] = viviendas['Fecha'].dt.year
    viviendas['Mes'] = viviendas['Fecha'].dt.month

    # Unir los conjuntos de datos
    df_final = pd.merge(viviendas, renta, how='left', left_on=['NMUN', 'Year'], right_on=['localidad_mapeada', 'Periodo'])

    # Crear datasets separados para 2021 y 2022
    data21 = df_final[df_final['Year'] == 2021]
    data22 = df_final[df_final['Year'] == 2022]

    # Imputar datos faltantes con la media para cada año
    renta_columns = [
        'Media de la renta por unidad de consumo',
        'Mediana de la renta por unidad de consumo',
        'Renta bruta media por hogar',
        'Renta bruta media por persona',
        'Renta neta media por hogar',
        'Renta neta media por persona '
    ]
    for col in renta_columns:
        data21[col] = data21[col].fillna(data21[col].mean())
        data22[col] = data22[col].fillna(data22[col].mean())

    # Eliminar columnas innecesarias
    data21 = data21.drop(['Municipios', 'Periodo', 'localidad_mapeada'], axis=1)
    data22 = data22.drop(['Municipios', 'Periodo', 'localidad_mapeada'], axis=1)

    # Concatenar datasets de 2021 y 2022
    df_final = pd.concat([data21, data22], ignore_index=True)

    # Guardar DataFrame limpios2 (opcional)
    if output_path:
        df_final.to_csv(output_path, index=False)

    return df_final


if __name__ == '__main__':
    viviendas_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosViviendasMadrid.csv'
    renta_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosRentaMadrid.csv'
    output_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosViviendasRentaMadrid.csv'

    cleaned_data = process_and_merge_data(viviendas_path, renta_path, output_path=output_path)