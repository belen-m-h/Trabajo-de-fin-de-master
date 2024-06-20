import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fuzzywuzzy import fuzz


def delete_missing_data(df):
    """
    Elimina columnas o filas con una gran cantidad de valores faltantes (más del 50%).

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame sin las columnas o filas con gran cantidad de valores faltantes.
    """
    umbral = 0.5 * len(df)
    df = df.dropna(thresh=umbral, axis=1)
    threshold = 14
    df = df[df.isna().sum(axis=1) <= threshold]
    return df


def data_imputation(df):
    """
    Imputa valores faltantes en el DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con los valores faltantes imputados.
    """
    print('Imputacion de datos: ')
    print('Caracteristicas nulas: ', df['Caracteristicas'].isnull().sum())
    df['Caracteristicas'] = df['Caracteristicas'].fillna('Desconocido')

    print('Moda de la columna Habitaciones: ', df['Habitaciones'].mode()[0])
    print('Habitaciones nulas: ', df['Habitaciones'].isnull().sum())
    df['Habitaciones'] = df['Habitaciones'].fillna(df['Habitaciones'].mode()[0])


    print('Moda de la columna Aseos: ', df['Aseos'].mode()[0])
    print('Aseos nulas: ', df['Aseos'].isnull().sum())
    df['Aseos'] = df['Aseos'].fillna(df['Aseos'].mode()[0])

    print('Mediana de la columna Precio: ', df['Precio'].median())
    print('Precio nulas: ', df['Precio'].isnull().sum())
    df['Precio'] = df['Precio'].fillna(df['Precio'].median())


    print('Mediana de la columna Metros: ', df['Metros'].median())
    print('Metros nulas: ', df['Metros'].isnull().sum())

    df['Metros'] = df['Metros'].fillna(df['Metros'].median())

    # Para las variables categoricas anadimos el valor desconocido
    for column in ['CodigoPostal', 'CMUN', 'CPRO', 'CCA', 'CUDIS', 'NPRO', 'NCA', 'NMUN']:
        df[column] = df[column].astype(str).fillna('Desconocido')

    return df


def fix_data_types(df):
    """
    Corrige los tipos de datos del DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con los tipos de datos corregidos.
    """
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)

    for column in ['Precio', 'Habitaciones', 'Aseos', 'Terraza', 'Piscina', 'Garaje']:
        df[column] = df[column].astype(int)

    return df


def fix_atypical_price_values(df):
    """
    Corrige valores atípicos en la columna 'Precio'.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con los valores atípicos corregidos o eliminados.
    """

    sns.histplot(df['Precio'], kde=True)
    plt.title('Distribución de Precios Antes de la Corrección de Valores Atípicos')
    plt.xlabel('Precio')
    plt.ylabel('Frecuencia')
    plt.show()

    Q1 = df['Precio'].quantile(0.25)
    Q3 = df['Precio'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 3 * IQR
    limite_superior = Q3 + 3 * IQR

    sns.histplot(df['Precio'], kde=True)
    plt.show()

    return df[(df['Precio'] > limite_inferior) & (df['Precio'] < limite_superior)]


def codification(df):
    """
    Convierte variables categóricas en variables ficticias.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con los datos codificados.
    """
    return pd.get_dummies(df, columns=['Características', 'CodigoPostal'])


def standarisation(df):
    """
    Estandariza las columnas 'Precio' y 'Metros' del DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con las columnas estandarizadas.
    """
    scaler = StandardScaler()
    df[['Precio', 'Metros']] = scaler.fit_transform(df[['Precio', 'Metros']])
    return df


def normalisation(df):
    """
    Normaliza las columnas 'Precio' y 'Metros' del DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con las columnas normalizadas.
    """
    min_max_scaler = MinMaxScaler()
    df[['Precio', 'Metros']] = min_max_scaler.fit_transform(df[['Precio', 'Metros']])
    return df


def unique_categorical_values(data):
    """
    Muestra los valores únicos de las columnas categóricas.

    Parámetros:
    - data (DataFrame): DataFrame a analizar.

    Devuelve:
    - None. Imprime los valores únicos en consola.
    """
    categorias = data.select_dtypes(include=['category', 'object']).columns
    for columna in categorias:
        print(f"Valores únicos en la columna '{columna}': {data[columna].unique()}")
    categorias_df = pd.DataFrame({
        'Columna': categorias,
        'Valores Únicos': [data[col].unique() for col in categorias]
    })
    print(categorias_df)


def fix_numerical_outliers(df):
    """
    Ajusta valores atípicos en las columnas numéricas del DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame a transformar.

    Devuelve:
    - DataFrame con valores atípicos ajustados.
    """
    df_fixed = df.copy()

    for column in ['Precio', 'Metros', 'Habitaciones', 'Aseos']:


        Q1 = df_fixed[column].quantile(0.25)
        Q3 = df_fixed[column].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 3 * IQR
        limite_superior = Q3 + 3 * IQR

        mediana = df_fixed[(df_fixed[column] >= limite_inferior) & (df_fixed[column] <= limite_superior)][
            column].median()
        df_fixed[column] = df_fixed[column].apply(
            lambda x: mediana if x < limite_inferior or x > limite_superior else x)

    return df_fixed


def load_and_preprocess_data(path):
    """
    Carga y preprocesa los datos iniciales desde un archivo CSV.
    Hacemos un cambio de tipo a algunas de las columnas para que sea el correcto.

    Parámetros:
    - path (str): Ruta del archivo CSV.

    Devuelve:
    - DataFrame con los datos cargados.
    """
    return pd.read_csv(
        path,
        delimiter=';',
        encoding='latin1',
        dtype={0: str, 16: str, 18: str}
    )


def filter_and_save_madrid_data(data, output_path):
    """
    Filtra los datos de la Comunidad de Madrid y los guarda en un archivo CSV.

    Parámetros:
    - data (DataFrame): DataFrame original.
    - output_path (str): Ruta del archivo CSV de salida.

    Devuelve:
    - DataFrame filtrado.
    """
    data = data[data['NCA'] == 'Comunidad de Madrid']
    data.to_csv(output_path, index=False)
    return data


def display_data_info(data):
    """
    Muestra información descriptiva del DataFrame.

    Parámetros:
    - data (DataFrame): DataFrame a analizar.

    Devuelve:
    - None. Imprime información en consola.
    """
    print(data.describe())
    print(data.info())
    print('Dimensiones de los datos: ', data.shape)


def drop_unnecessary_columns(data, columns_to_drop):
    """
    Elimina columnas innecesarias del DataFrame.

    Parámetros:
    - data (DataFrame): DataFrame original.
    - columns_to_drop (list): Lista de nombres de columnas a eliminar.

    Devuelve:
    - DataFrame con las columnas eliminadas.
    """
    return data.drop(columns=columns_to_drop)


def clean_data(path, delimiter=';', encoding='latin1', output_path=None):
    """
    Limpia y procesa los datos del archivo de viviendas CSV.

    Parámetros:
    - path (str): Ruta del archivo CSV.
    - delimiter (str): Delimitador para separar las columnas.
    - encoding (str): Codificación del archivo.
    - output_path (str, opcional): Ruta del archivo CSV para guardar los datos limpios.

    Devuelve:
    - DataFrame con los datos limpios.
    """
    data = pd.read_csv(
        path,
        delimiter=delimiter,
        encoding=encoding,
        dtype={0: str, 16: str, 18: str}
    )

    # Para obtener unicamente los datos de la comunidad
    # de madrid filtramos por el valor correspondiente
    data = data[data['NCA'] == 'Comunidad de Madrid']


    # Filtrar los datos solo para las comprar
    data = data[~data['URL'].str.contains(r'/alquiler/', regex=True, case=False)]

    # Mostramos informacion descriptiva de los datos
    display_data_info(data)

    # Eliminamos aquellas columnas o filas con mayoria de datos faltantes
    data = delete_missing_data(data)

    # Tras inspeccionar los datos eliminamos algunas columnas que no nos
    # aportaran informacion importante a la hora de predecir precios
    columns_to_drop = ['Relacion', 'Precision', 'Inmueble', 'ID', 'URL', 'ID_Cliente', 'URL_Cliente', 'Unnamed: 0']
    data = data.drop(columns=columns_to_drop)

    num_filas_con_nulos = data.isnull().any(axis=1).sum()

    print(f"Número de filas con algún valor nulo: {num_filas_con_nulos}")
    # Para seguir tratando los datos faltantes imputamos algunos de los valores
    data = data_imputation(data)

    # Corregimos el tipo de datos de las columnas pertinentes
    data = fix_data_types(data)

    # Comprobamos los valores de los datos categoricos para comprobar anomalias
    unique_categorical_values(data)

    # Tras la exploracion eliminamos algunas variables que no nos aportan informacion importante
    columns_to_drop = ['NPRO', 'NCA', 'CMUN', 'CPRO', 'CCA', 'CUDIS']
    data = data.drop(columns=columns_to_drop)

    # Eimputamos el valor nan observado en la columna CodigoPostal
    data['CodigoPostal'] = data['CodigoPostal'].replace('nan', 'Desconocido')

    # Eliminamos valores outlier de las variables numericas
    data = fix_numerical_outliers(data)

    # Guardamos los datos en el path especificado si existe
    if output_path:
        data.to_csv(output_path, index=False)

    return data


if __name__ == '__main__':
    # Path de los datos de viviendas de los que disponemos
    path = '../../data/DatosViviendas.csv'

    # Path para guardar unicamente los datos de Madrid
    output_path = '../data/procesados/DatosViviendasMadrid.csv'

    clean_data(path, output_path=output_path)
