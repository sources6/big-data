import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

st.title("Peleas Ganadas por Peleadores con Precisión >50%")

# Conectar a la base de datos
conn = sqlite3.connect("mi_base_de_datos.db")

# Crear la consulta SQL
query = """
SELECT
    `Fighter1`, `Fighter2`,
    `Total Strike Landed F1R1`, `Total Strike Landed F1R2`, `Total Strike Landed F1R3`,
    `Total Strike Missed F1R1`, `Total Strike Missed F1R2`, `Total Strike Missed F1R3`,
    `Total Strike Landed F2R1`, `Total Strike Landed F2R2`, `Total Strike Landed F2R3`,
    `Total Strike Missed F2R1`, `Total Strike Missed F2R2`, `Total Strike Missed F2R3`,
    `Ctrl Time (Minutes) F1R1`, `Ctrl Time (Minutes) F2R1`,
    `Winner?`
FROM datos
"""
df = pd.read_sql(query, conn)

# Calcular la precisión de golpes
for fighter in ["F1", "F2"]:
    df[f'Accuracy_{fighter}'] = (df[f'Total Strike Landed {fighter}R1'] +
                                  df[f'Total Strike Landed {fighter}R2'] +
                                  df[f'Total Strike Landed {fighter}R3']) / (
                                 (df[f'Total Strike Landed {fighter}R1'] + df[f'Total Strike Missed {fighter}R1']) +
                                 (df[f'Total Strike Landed {fighter}R2'] + df[f'Total Strike Missed {fighter}R2']) +
                                 (df[f'Total Strike Landed {fighter}R3'] + df[f'Total Strike Missed {fighter}R3'])
                                )

df.dropna(subset=["Accuracy_F1", "Accuracy_F2"], inplace=True)

df["High_Accuracy_F1"] = df["Accuracy_F1"] >= 0.5
df["High_Accuracy_F2"] = df["Accuracy_F2"] >= 0.5
df["High_Accuracy_Winner_F1"] = df["High_Accuracy_F1"] & (df["Winner?"] == 1)
df["High_Accuracy_Winner_F2"] = df["High_Accuracy_F2"] & (df["Winner?"] == 0)

# Cálculos
high_acc_wins_f1 = df["High_Accuracy_Winner_F1"].sum()
high_acc_wins_f2 = df["High_Accuracy_Winner_F2"].sum()
total_fights = len(df)
high_acc_total = high_acc_wins_f1 + high_acc_wins_f2

st.write(f"Peleadores con precisión >50% ganaron: {high_acc_total} peleas ({(high_acc_total / total_fights) * 100:.2f}%)")

# Crear la gráfica
fig, ax = plt.subplots()
fighters = ['F1', 'F2']
wins = [high_acc_wins_f1, high_acc_wins_f2]
win_percentage = [(high_acc_wins_f1 / total_fights) * 100, (high_acc_wins_f2 / total_fights) * 100]

ax.bar(fighters, wins, color='skyblue', label='Peleas Ganadas')
ax.set_xlabel('Peleador')
ax.set_ylabel('Número de Peleas Ganadas')
ax.set_title('Peleas Ganadas por Peleadores con Precisión >50%')

for i in range(len(fighters)):
    ax.text(i, wins[i] + 1, f'{win_percentage[i]:.2f}%', ha='center', va='bottom')

plt.tight_layout()
st.pyplot(fig)



# Título de la app
st.title("Métodos de Victoria en Peleas Rápidas")

# Conectar a la base de datos SQLite
conn = sqlite3.connect("mi_base_de_datos.db")

# Consulta SQL para obtener los datos necesarios
query = """
SELECT `Fighter1`, `Fighter2`, `Time`, `Fight Method`
FROM datos
"""

# Leer los datos en un DataFrame
df = pd.read_sql(query, conn)

# Cerrar la conexión
conn.close()

# Convertir la columna 'Time' a formato timedelta
try:
    df['Time'] = pd.to_timedelta(df['Time'])
except ValueError:
    df['Time'] = pd.to_timedelta(df['Time'], errors='coerce')

# Eliminar filas con valores nulos en la columna 'Time' (si hay errores de conversión)
df.dropna(subset=['Time'], inplace=True)

# Convertir 'Time' a segundos
df['Time_seconds'] = df['Time'].dt.total_seconds()

# Filtrar peleas que terminaron en menos de 5 minutos (300 segundos)
df_peleas_rapidas = df[df['Time_seconds'] <= 300]

# Filtrar peleas que terminaron por KO/TKO en menos de 5 minutos
ko_tko_peleas_rapidas = df_peleas_rapidas[df_peleas_rapidas['Fight Method'] == 'KO/TKO']

# Contar el total de peleas rápidas y las victorias por KO/TKO
total_peleas_rapidas = len(df_peleas_rapidas)
ko_tko_peleas_rapidas_count = len(ko_tko_peleas_rapidas)

# Calcular el porcentaje de peleas rápidas terminadas por KO/TKO
ko_tko_percentage = (ko_tko_peleas_rapidas_count / total_peleas_rapidas) * 100 if total_peleas_rapidas > 0 else 0

# Mostrar los resultados en Streamlit
st.write(f"Total de peleas terminadas rápidamente (<= 05:00): {total_peleas_rapidas}")
st.write(f"Peleas terminadas por KO/TKO en menos de 05:00: {ko_tko_peleas_rapidas_count}")
st.write(f"Porcentaje de peleas terminadas por KO/TKO en peleas rápidas: {ko_tko_percentage:.2f}%")

# Verificar si la hipótesis se confirma
if ko_tko_percentage >= 70:
    st.write("La hipótesis se confirma: El 70% de las peleas rápidas son por KO/TKO.")
else:
    st.write("La hipótesis no se confirma: El porcentaje de peleas rápidas por KO/TKO es menor al 70%.")

# Contar el total de peleas por cada método de victoria en peleas rápidas
victorias_por_metodo = df_peleas_rapidas['Fight Method'].value_counts()

# Crear la gráfica de barras
fig, ax = plt.subplots(figsize=(8, 6))
victorias_por_metodo.plot(kind='bar', color='skyblue', ax=ax)

# Personalizar el gráfico
ax.set_title('Métodos de Victoria en Peleas Rápidas (menos de 5 minutos)', fontsize=14)
ax.set_xlabel('Método de Victoria', fontsize=12)
ax.set_ylabel('Número de Peleas', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)


# Conectar a la base de datos SQLite
conn = sqlite3.connect("mi_base_de_datos.db")

# Consulta SQL para obtener los datos necesarios
query = """
SELECT `Fighter1`, `Fighter2`, `Total Strike Landed F1R1`, `Total Strike Landed F1R2`,
       `Total Strike Landed F2R1`, `Total Strike Landed F2R2`, `Winner?`
FROM datos
"""

# Leer los datos en un DataFrame
df = pd.read_sql(query, conn)

# Cerrar la conexión
conn.close()

# Sumar los golpes conectados en los primeros dos rounds para cada peleador
df['Total_Strikes_F1_R1_R2'] = df['Total Strike Landed F1R1'] + df['Total Strike Landed F1R2']
df['Total_Strikes_F2_R1_R2'] = df['Total Strike Landed F2R1'] + df['Total Strike Landed F2R2']

# Clasificar el resultado de la pelea: 1 = Peleador 1 gana, 0 = Peleador 2 gana
df['F1_Wins'] = df['Winner?'] == 1

# Comparar los golpes conectados por cada peleador con el resultado de la pelea
df['F1_Higher_Strikes'] = df['Total_Strikes_F1_R1_R2'] > df['Total_Strikes_F2_R1_R2']

# Ver cuántas veces el peleador con más golpes en los primeros dos rounds ganó
correct_predictions = df['F1_Higher_Strikes'] == df['F1_Wins']

# Calcular el porcentaje de veces que el peleador con más golpes en los primeros dos rounds ganó
accuracy = correct_predictions.sum() / len(df) * 100

# Mostrar el resultado en Streamlit
st.write(f'El porcentaje de veces que el peleador con más golpes en los primeros dos rounds ganó es: {accuracy:.2f}%')

# Mostrar los primeros registros para verificar si el cálculo es correcto
st.write(df[['Fighter1', 'Fighter2', 'Total_Strikes_F1_R1_R2', 'Total_Strikes_F2_R1_R2', 'Winner?', 'F1_Wins', 'F1_Higher_Strikes']].head())

# Crear la gráfica con Seaborn
plt.figure(figsize=(10, 6))

# Gráfico de dispersión con los golpes conectados en los primeros dos rounds en el eje X
# y el resultado de la pelea (1 = victoria del peleador 1, 0 = victoria del peleador 2) en el eje Y
sns.scatterplot(x='Total_Strikes_F1_R1_R2', y='Total_Strikes_F2_R1_R2', hue='F1_Wins', data=df, palette='coolwarm', alpha=0.6)

# Personalizar la gráfica
plt.title('Comparación de Golpes Conectados en los Primeros 2 Rounds con Resultado de')


# Conectar a la base de datos SQLite
conn = sqlite3.connect("mi_base_de_datos.db")

# Consulta SQL para obtener los datos necesarios
query = """
SELECT `Fighter1`, `Fighter2`, `Total Strike Landed F1R1`, `Total Strike Landed F1R2`,
       `Total Strike Landed F2R1`, `Total Strike Landed F2R2`, `Winner?`
FROM datos
"""

# Leer los datos en un DataFrame
df = pd.read_sql(query, conn)

# Cerrar la conexión
conn.close()

# Sumar los golpes conectados en los primeros dos rounds para cada peleador
df['Total_Strikes_F1_R1_R2'] = df['Total Strike Landed F1R1'] + df['Total Strike Landed F1R2']
df['Total_Strikes_F2_R1_R2'] = df['Total Strike Landed F2R1'] + df['Total Strike Landed F2R2']

# Clasificar el resultado de la pelea: 1 = Peleador 1 gana, 0 = Peleador 2 gana
df['F1_Wins'] = df['Winner?'] == 1

# Comparar los golpes conectados por cada peleador con el resultado de la pelea
df['F1_Higher_Strikes'] = df['Total_Strikes_F1_R1_R2'] > df['Total_Strikes_F2_R1_R2']

# Ver cuántas veces el peleador con más golpes en los primeros dos rounds ganó
correct_predictions = df['F1_Higher_Strikes'] == df['F1_Wins']

# Calcular el porcentaje de veces que el peleador con más golpes en los primeros dos rounds ganó
accuracy = correct_predictions.sum() / len(df) * 100

# Mostrar el resultado en Streamlit
st.write(f'El porcentaje de veces que el peleador con más golpes en los primeros dos rounds ganó es: {accuracy:.2f}%')

# Mostrar los primeros registros para verificar si el cálculo es correcto
st.write(df[['Fighter1', 'Fighter2', 'Total_Strikes_F1_R1_R2', 'Total_Strikes_F2_R1_R2', 'Winner?', 'F1_Wins', 'F1_Higher_Strikes']].head())

# Crear la gráfica con Seaborn
plt.figure(figsize=(10, 6))

# Gráfico de dispersión con los golpes conectados en los primeros dos rounds en el eje X
# y el resultado de la pelea (1 = victoria del peleador 1, 0 = victoria del peleador 2) en el eje Y
sns.scatterplot(x='Total_Strikes_F1_R1_R2', y='Total_Strikes_F2_R1_R2', hue='F1_Wins', data=df, palette='coolwarm', alpha=0.6)

# Personalizar la gráfica
plt.title('Comparación de Golpes Conectados en los Primeros 2 Rounds con Resultado de la Pelea')
plt.xlabel('Golpes Conectados por Peleador 1 en Rounds 1 y 2')
plt.ylabel('Golpes Conectados por Peleador 2 en Rounds 1 y 2')
plt.legend(title='Victoria del Peleador 1', loc='upper left')

# Mostrar la gráfica en Streamlit
st.pyplot(plt)


# Conectar a la base de datos SQLite
conn = sqlite3.connect("mi_base_de_datos.db")

# Consulta SQL para obtener los datos necesarios
query = """
SELECT `Fighter1`, `Fighter2`, `TD Completed F1R1`, `TD Completed F2R1`, `Winner?`
FROM datos
"""

# Leer los datos en un DataFrame
df = pd.read_sql(query, conn)

# Cerrar la conexión
conn.close()

# Hipótesis: El peleador con más derribos completados en el primer round tiene más probabilidades de ganar

# Renombrar columnas para facilitar el manejo
df.rename(columns={
    'TD Completed F1R1': 'Derribo_F1_R1',
    'TD Completed F2R1': 'Derribo_F2_R1'
}, inplace=True)

# Crear una nueva columna que nos diga qué peleador completó más derribos en el primer round
df['Derribo_Mayor'] = df.apply(lambda row: 'F1' if row['Derribo_F1_R1'] > row['Derribo_F2_R1'] else 'F2', axis=1)

# Clasificar el resultado de la pelea: 1 = Peleador 1 gana, 0 = Peleador 2 gana
df['F1_Wins'] = df['Winner?'] == 1

# Verificamos si el peleador con más derribos completados ganó
df['Resultado_Comprobacion'] = df['Derribo_Mayor'] == df['F1_Wins'].apply(lambda x: 'F1' if x else 'F2')

# Calcular el porcentaje de aciertos para la hipótesis
porcentaje_aciertos = df['Resultado_Comprobacion'].mean() * 100
st.write(f"Porcentaje de aciertos para la hipótesis: {porcentaje_aciertos:.2f}%")

# Mostrar los primeros registros para verificar si el cálculo es correcto
st.write(df[['Fighter1', 'Fighter2', 'Derribo_F1_R1', 'Derribo_F2_R1', 'Winner?', 'F1_Wins', 'Resultado_Comprobacion']].head())

# Crear gráfico de dispersión
plt.figure(figsize=(10, 6))

# Gráfico de dispersión con los derribos completados en el primer round de ambos peleadores
sns.scatterplot(x='Derribo_F1_R1', y='Derribo_F2_R1', hue='F1_Wins', data=df, palette='coolwarm', alpha=0.6)

# Personalizar
plt.title('Derribos Completados en el Primer Round vs Resultado de la Pelea')
plt.xlabel('Derribos Completados Peleador 1')
plt.ylabel('Derribos Completados Peleador 2')
plt.legend(title='Resultado de la Pelea', loc='upper left')

# Mostrar el gráfico en Streamlit
st.pyplot(plt)


# Conectar a la base de datos SQLite
conn = sqlite3.connect("mi_base_de_datos.db")

# Consulta SQL para obtener los datos necesarios
query = """
SELECT 
    `Fighter1`, 
    `Fighter2`, 
    `Ctrl Time (Minutes) F1R1`, 
    `Ctrl Time (Minutes) F2R1`, 
    `Winner?`
FROM datos
"""

# Leer los datos en un DataFrame
df = pd.read_sql(query, conn)

# Cerrar la conexión
conn.close()

# Clasificar el resultado de la pelea: 1 = Peleador 1 gana, 0 = Peleador 2 gana
df['F1_Wins'] = df['Winner?'] == 1

# Filtrar las columnas de tiempo de control en el primer round para cada peleador desde la base de datos
df['Ctrl_Time_F1_R1'] = df['Ctrl Time (Minutes) F1R1']
df['Ctrl_Time_F2_R1'] = df['Ctrl Time (Minutes) F2R1']

# Crear una nueva columna que nos diga qué peleador tuvo más control en el suelo en el primer round
df['Control_Mayor'] = df.apply(lambda row: 'F1' if row['Ctrl_Time_F1_R1'] > row['Ctrl_Time_F2_R1'] else 'F2', axis=1)

# Verificamos si el peleador con mayor control en el suelo ganó
df['Resultado_Comprobacion'] = df['Control_Mayor'] == df['F1_Wins'].apply(lambda x: 'F1' if x == 1 else 'F2')

# Calcular el porcentaje de aciertos para la hipótesis
porcentaje_aciertos = df['Resultado_Comprobacion'].mean() * 100
st.write(f"Porcentaje de aciertos para la hipótesis: {porcentaje_aciertos:.2f}%")

# Mostrar los primeros registros para verificar si el cálculo es correcto
st.write(df[['Fighter1', 'Fighter2', 'Ctrl_Time_F1_R1', 'Ctrl_Time_F2_R1', 'Winner?', 'F1_Wins', 'Resultado_Comprobacion']].head())

# Crear gráfico de dispersión
plt.figure(figsize=(10, 6))

# Gráfico de dispersión con el control en el suelo del primer round de ambos peleadores
sns.scatterplot(x='Ctrl_Time_F1_R1', y='Ctrl_Time_F2_R1', hue='F1_Wins', data=df, palette='coolwarm', alpha=0.6)

# Personalizar la gráfica
plt.title('Comparación de Control en el Suelo en el Primer Round con el Resultado de la Pelea')
plt.xlabel('Tiempo de Control en el Suelo de Peleador 1 (Primer Round)')
plt.ylabel('Tiempo de Control en el Suelo de Peleador 2 (Primer Round)')
plt.legend(title='Victoria del Peleador 1', loc='upper left')

# Mostrar el gráfico en Streamlit
st.pyplot(plt)


