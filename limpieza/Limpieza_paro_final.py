import pandas as pd


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
        'total Paro Registrado',
        'Paro hombre edad < 25',
        'Paro hombre edad 25 -45 ',
        'Paro hombre edad >=45',
        'Paro mujer edad < 25',
        'Paro mujer edad 25 -45 ',
        'Paro mujer edad >=45',
        'Paro Agricultura',
        'Paro Industria',
        'Paro construccion',
        'Paro Servicios',
        'Paro Sin empleo Anterior',
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


def check_missing_percentage(df):
    """
    Calcula y muestra el porcentaje de valores faltantes en cada columna del DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a analizar.

    Devuelve:
    - Series con el porcentaje de valores faltantes por columna.
    """
    missing_percentage = (df.isna().sum() / len(df)) * 100
    print(missing_percentage)
    return missing_percentage


def rename_columns(df, columns_dict):
    """
    Renombra columnas en el DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.
    - columns_dict (dict): Diccionario con los nombres actuales y los nuevos nombres de las columnas.

    Devuelve:
    - DataFrame con las columnas renombradas.
    """
    df.rename(columns=columns_dict, inplace=True)
    return df


def filter_by_region(df, region):
    """
    Filtra el DataFrame por una región específica.

    Parámetros:
    - df (DataFrame): DataFrame a filtrar.
    - region (str): Región por la cual se desea filtrar.

    Devuelve:
    - DataFrame filtrado por la región especificada.
    """
    return df[df['Comunidad Autonoma'] == region]


def process_dataframe(df, region):
    """
    Procesa el DataFrame: filtra por región, elimina columnas innecesarias, y extrae mes y año de 'Codigo Mes'.

    Parámetros:
    - df (DataFrame): DataFrame a procesar.
    - region (str): Región por la cual se desea filtrar.

    Devuelve:
    - DataFrame procesado.
    """
    df = filter_by_region(df, region)
    df = df.drop(['Codigo de CA', 'Comunidad Autonoma', 'Codigo Provincia', 'Provincia', 'Codigo Municipio', 'mes'],
                 axis=1)
    df['Codigo Mes'] = df['Codigo Mes'].astype(str)
    df['mes'] = df['Codigo Mes'].str[-2:]
    df['year'] = df['Codigo Mes'].str[:4]
    df = df.drop(['Codigo Mes'], axis=1)
    return df


def clean_and_process_data(paro21_path, paro22_path, output_path=None):
    """
    Limpia y procesa los datos de paro de dos archivos CSV para los años 2021 y 2022.

    Parámetros:
    - paro21_path (str): Ruta del archivo CSV de 2021.
    - paro22_path (str): Ruta del archivo CSV de 2022.
    - region (str): Región por la cual se desea filtrar.
    - output_path (str, opcional): Ruta del archivo CSV para guardar los datos combinados.

    Devuelve:
    - DataFrame con los datos combinados y procesados.
    """
    # Crear DataFrames para 2021 y 2022
    paro21 = create_dataframe(paro21_path, ";", "latin1")
    paro22 = create_dataframe(paro22_path, ";", "latin1")
    region = 'Madrid, Comunidad de'

    # Mostrar primeros registros y columnas
    print('DataFrame de Paro 2021:')
    print(paro21.head())
    print(paro21.columns)

    # Calcular y mostrar porcentaje de valores faltantes
    print("Porcentaje de valores faltantes en Paro 2021:")
    check_missing_percentage(paro21)

    # Renombrar columnas
    columns_dict = {
        'Cï¿½digo mes ': 'Codigo Mes',
        'Cï¿½digo de CA': 'Codigo de CA',
        'Paro Construcciï¿½n': 'Paro construccion',
        'Comunidad Autï¿½noma': 'Comunidad Autonoma'
    }
    paro21 = rename_columns(paro21, columns_dict)

    # Procesar DataFrame de 2021
    paro21 = process_dataframe(paro21, region)
    paro21 = fix_numerical_outliers(paro21)
    # Guardar DataFrame de 2021 filtrado (opcional)

    print('DataFrame de Paro 2021 procesado:')
    print(paro21.head())

    # Mostrar primeros registros y columnas
    print('DataFrame de Paro 2022:')
    print(paro22.head())
    print(paro22.columns)

    # Calcular y mostrar porcentaje de valores faltantes
    print("Porcentaje de valores faltantes en Paro 2022:")
    check_missing_percentage(paro22)

    # Renombrar columnas
    paro22 = rename_columns(paro22, columns_dict)

    # Procesar DataFrame de 2022
    paro22 = process_dataframe(paro22, region)
    columns_to_convert = [
        'total Paro Registrado',
        'Paro hombre edad < 25',
        'Paro hombre edad 25 -45 ',
        'Paro hombre edad >=45',
        'Paro mujer edad < 25',
        'Paro mujer edad 25 -45 ',
        'Paro mujer edad >=45',
        'Paro Agricultura',
        'Paro Industria',
        'Paro construccion',
        'Paro Servicios',
        'Paro Sin empleo Anterior',
    ]
    # Convertir las columnas a int
    for column in columns_to_convert:
        paro22[column] = paro22[column].str.replace('[^\d.]', '', regex=True)
        paro22[column] = paro22[column].astype('float64')

    paro22 = fix_numerical_outliers(paro22)

    # Guardar DataFrame de 2022 filtrado (opcional)
    print('DataFrame de Paro 2022 procesado:')
    print(paro22.head())

    # Concatenar DataFrames de 2021 y 2022
    df_concatenado = pd.concat([paro21, paro22], ignore_index=True)

    # Guardar DataFrame concatenado (opcional)
    if output_path:
        df_concatenado.to_csv(output_path, index=False)

    return df_concatenado


if __name__ == '__main__':
    paro21_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/Paro_por_municipios_2021_csv.csv'
    paro22_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/Paro_por_municipios_2022_csv.csv'

    output_path = '/Users/belenmunozhernandez/PycharmProjects/pythonProject/data/procesados/DatosParoMadrid.csv'

    cleaned_data = clean_and_process_data(paro21_path, paro22_path, output_path=output_path)
