import pandas as pd
import numpy as np


def fix_numerical_outliers(df):
    """
    Ajusta valores atípicos en las columnas numéricas del DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con valores atípicos ajustados.
    """
    df_fixed = df.copy()

    for column in [
        'Media de la renta por unidad de consumo',
        'Mediana de la renta por unidad de consumo',
        'Renta bruta media por hogar',
        'Renta bruta media por persona',
        'Renta neta media por hogar',
        'Renta neta media por persona ',
    ]:
        Q1 = df_fixed[column].quantile(0.25)
        Q3 = df_fixed[column].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 3 * IQR
        limite_superior = Q3 + 3 * IQR


        df_fixed[column] = df_fixed[column].apply(
            lambda x: None if x < limite_inferior or x > limite_superior else x)
        df_fixed = df_fixed.dropna()
    return df_fixed


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


def filter_data_for_year(df, year):
    """
    Filtra el DataFrame para un año específico.

    Parámetros:
    - df (DataFrame): DataFrame a filtrar.
    - year (int): Año para filtrar los datos.

    Devuelve:
    - DataFrame filtrado para el año especificado.
    """
    return df[df['Periodo'].isin([year])]


def rename_column(df, old_name, new_name):
    """
    Renombra una columna en el DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame.
    - old_name (str): Nombre actual de la columna.
    - new_name (str): Nuevo nombre de la columna.

    Devuelve:
    - DataFrame con la columna renombrada.
    """
    df.rename(columns={old_name: new_name}, inplace=True)
    return df


def pivot_dataframe(df):
    """
    Crea un DataFrame pivoteado basado en 'Municipios' y 'Periodo'.

    Parámetros:
    - df (DataFrame): DataFrame original.

    Devuelve:
    - DataFrame pivoteado.
    """
    pivot_df = df.pivot_table(
        index=['Municipios', 'Periodo'],
        columns='Indicadores de renta media y mediana',
        values='Total',
        aggfunc='first'
    )
    pivot_df.reset_index(inplace=True)
    return pivot_df


def adjust_for_inflation(df, columns, inflation_rate, year):
    """
    Ajusta los valores de renta para un nuevo año basado en la inflación.

    Parámetros:
    - df (DataFrame): DataFrame original.
    - columns (list): Lista de columnas a ajustar.
    - inflation_rate (float): Tasa de inflación (decimal).
    - year (int): Año nuevo para los datos ajustados.

    Devuelve:
    - DataFrame con los valores ajustados para la inflación.
    """
    df_adjusted = df.copy()
    df_adjusted[columns] = df_adjusted[columns] * (1 + inflation_rate)
    df_adjusted['Periodo'] = year
    return df_adjusted


def clean_and_process_data(path, delimiter=';', encoding='latin1', output_path=None):
    """
    Limpia y procesa los datos del archivo CSV de renta media por persona.

    Parámetros:
    - path (str): Ruta del archivo CSV.
    - delimiter (str): Delimitador para separar las columnas.
    - encoding (str): Codificación del archivo.
    - output_path (str, opcional): Ruta del archivo CSV para guardar los datos limpios.

    Devuelve:
    - DataFrame con los datos limpios.
    """
    # Creamos el DataFrame a partir del archivo CSV
    renta = create_dataframe(path, delimiter, encoding)

    # Filtrar los datos para el año 2021, ya que es el unico año que nos interesa de los disponibles
    renta = filter_data_for_year(renta, 2021)

    # Paso 3: Renombrar la columna para mayor legibilidad
    renta = rename_column(renta, 'ï»¿Municipios', 'Municipios')

    print(renta['Indicadores de renta media y mediana'].unique())
    # Creamos un DataFrame pivoteado para obtener todos los datos de renta por municipio y periodo
    renta = pivot_dataframe(renta)

    #  Ajustamos los datos para el año 2022 basados en la inflación
    inflation_rate = 0.049  # Tasa de inflación del 4.9%
    renta_columns = renta.columns.difference(['Periodo', 'Municipios'])
    renta[renta_columns] = renta[renta_columns].apply(pd.to_numeric, errors='coerce')
    renta_2022 = adjust_for_inflation(renta, renta_columns, inflation_rate, 2022)

    # Combinamos los datos de 2021 y 2022
    df_combined = pd.concat([renta, renta_2022], ignore_index=True)

    # Corregimos los outliers de los valores numéricos
    df_combined = fix_numerical_outliers(df_combined)

    # Finalmente guardamos los datos combinados
    if output_path:
        df_combined.to_csv(output_path, index=False)

    return df_combined


if __name__ == '__main__':
    path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/RentaMediaPorPersona.csv'
    output_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosRentaMadrid.csv'

    cleaned_data = clean_and_process_data(path, output_path=output_path)
