# Predicción del Mercado Inmobiliario en la Comunidad Autónoma de Madrid Utilizando Aprendizaje Automático y Datos Abiertos
## Autor
Belén Muñoz Hernández

## Tutor
Jorge Segura Gisbert

## Universidad 
Universidad Oberta de Catalunya
## Resumen
Este trabajo tiene como propósito el estudio y creación de modelos predictivos orientados a estimar los precios de la vivienda en la Comunidad Autónoma de Madrid. Para ello se utilizarán técnicas de Machine Learning, entre otras que puedan resultar interesantes, sobre un conjunto amplio de datos abiertos proporcionado por un portal inmobiliario. Estos datos se nutrirán  de otras fuentes de datos abiertas, para tener en cuenta otras variables que puedan ser de utilidad, como por ejemplo la renta media de los hogares en cada municipio. Para este objetivo, se propone abordar el uso de herramientas avanzadas de análisis de datos y técnicas de aprendizaje automático en este ámbito.

## Contenido
1. [Datos](#datos)
2. [Limpieza](#limpieza)
3. [Modelos](#modelos)


## Datos
- DatosCompletos.csv: Contiene todos los datos finales.
- DatosParoMadrid.csv: Datos de paro en Madrid.
- DatosRentaMadrid.csv: Datos de renta en Madrid.
- DatosViviendasMadrid.csv: Datos de viviendas en Madrid.
- DatosViviendasRentaMadrid.csv: Datos combinados de viviendas y renta en Madrid.
- DatosViviendas.csv.zip: Archivo comprimido con los datos de viviendas.
- Paro_por_municipios_2021_csv.csv: Datos de paro por municipios en 2021.
- Paro_por_municipios_2022_csv.csv: Datos de paro por municipios en 2022.
- RentaMediaPorPersona.csv: Datos de renta media por persona.

## Limpieza
- Limpieza_viviendas: procesamiento de los datos de viviendas.
- Limpieza_paro: procesamiento de los datos de paro.  
- Limpieza_renta: procesamiento de los datos de renta.  
- Juntar_viviendas_renta: integración de los datos de vivienda y renta
- Juntar_viviendas_renta_con_paro: unir al conjunto de viviendas con renta los datos de paro

## Modelos

Modelos empleados en el trabajo en formato notebook ipynb.

- ann: Notebook con los resultados del modelo de redes neuronales.
- decisiontreeregressor: Notebook con los resultados del modelo Decision Tree Regressor.
- gradient-boosting-regressor: Notebook con los resultados del modelo gradient boosting regressor.
- rand-forest: Notebook con los resultados del modelo rendom forest.
- xgboost: Notebook con los resultados del modelo XGboost.
