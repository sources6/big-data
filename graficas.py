# la precisión de golpes es factor clave para la victoria, el peleador el cual conecta el 50% de sus ataques(golpes y patadas) tiene mayor probabilidad de ganar una pelea, se comprobará la tasa de victorias de peleadores con precisión alta vs baja
# Calcular la precisión de golpes para cada peleador
for fighter in ["F1", "F2"]:
    df[f'Accuracy_{fighter}'] = (df[f'Total Strike Landed {fighter}R1'] +
                                  df[f'Total Strike Landed {fighter}R2'] +
                                  df[f'Total Strike Landed {fighter}R3']) / (
                                 (df[f'Total Strike Landed {fighter}R1'] + df[f'Total Strike Missed {fighter}R1']) +
                                 (df[f'Total Strike Landed {fighter}R2'] + df[f'Total Strike Missed {fighter}R2']) +
                                 (df[f'Total Strike Landed {fighter}R3'] + df[f'Total Strike Missed {fighter}R3'])
                                )

# Verificar si hay valores nulos en la precisión
df.dropna(subset=["Accuracy_F1", "Accuracy_F2"], inplace=True)

# Clasificar peleadores con precisión alta (+50%) y baja (-50%)
df["High_Accuracy_F1"] = df["Accuracy_F1"] >= 0.5
df["High_Accuracy_F2"] = df["Accuracy_F2"] >= 0.5

# Asumimos que el ganador está indicado por el valor 1 en 'Winner?'
# Cambiar "1" por el nombre real del ganador si es necesario.
df["High_Accuracy_Winner_F1"] = df["High_Accuracy_F1"] & (df["Winner?"] == 1)  # Si F1 ganó y tiene alta precisión
df["High_Accuracy_Winner_F2"] = df["High_Accuracy_F2"] & (df["Winner?"] == 0)  # Si F2 ganó y tiene alta precisión

# Calcular las victorias de los peleadores con alta precisión
high_acc_wins_f1 = df["High_Accuracy_Winner_F1"].sum()
high_acc_wins_f2 = df["High_Accuracy_Winner_F2"].sum()

# Calcular la tasa de victoria de los peleadores con alta precisión
total_fights = len(df)
high_acc_total = high_acc_wins_f1 + high_acc_wins_f2

# Mostrar el total de peleas ganadas por los peleadores con alta precisión
print(f"Peleadores con precisión >50% ganaron: {high_acc_total} peleas ({(high_acc_total / total_fights) * 100:.2f}%)")

# Crear la gráfica
fig, ax = plt.subplots()

# Datos para las barras
fighters = ['F1', 'F2']
wins = [high_acc_wins_f1, high_acc_wins_f2]
win_percentage = [(high_acc_wins_f1 / total_fights) * 100, (high_acc_wins_f2 / total_fights) * 100]

# Graficar barras
ax.bar(fighters, wins, color='skyblue', label='Peleas Ganadas')
ax.set_xlabel('Peleador')
ax.set_ylabel('Número de Peleas Ganadas')
ax.set_title('Peleas Ganadas por Peleadores con Precisión >50%')

# Mostrar porcentaje encima de las barras
for i in range(len(fighters)):
    ax.text(i, wins[i] + 5, f'{win_percentage[i]:.2f}%', ha='center', va='bottom')

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# También podemos ver el resultado de las primeras filas para ver si todo se calculó correctamente
print(df[["Fighter1", "Fighter2", "Accuracy_F1", "Accuracy_F2", "Winner?", "High_Accuracy_F1", "High_Accuracy_F2", "High_Accuracy_Winner_F1", "High_Accuracy_Winner_F2"]].head())

#********


# Supongamos que ya tienes el DataFrame 'df' cargado y que la columna 'Time' está en formato 'datetime.time'

# Convertir la columna 'Time' a segundos
df['Time_seconds'] = df['Time'].apply(lambda x: x.minute * 60 + x.second)

# Filtrar peleas que terminaron en menos de 5 minutos (300 segundos)
df_peleas_rapidas = df[df['Time_seconds'] <= 300]

# Filtrar peleas que terminaron por KO/TKO en menos de 5 minutos
ko_tko_peleas_rapidas = df_peleas_rapidas[df_peleas_rapidas['Fight Method'] == 'KO/TKO']

# Contar el total de peleas rápidas y las victorias por KO/TKO
total_peleas_rapidas = len(df_peleas_rapidas)
ko_tko_peleas_rapidas_count = len(ko_tko_peleas_rapidas)

# Calcular el porcentaje de peleas rápidas terminadas por KO/TKO
ko_tko_percentage = (ko_tko_peleas_rapidas_count / total_peleas_rapidas) * 100

# Mostrar los resultados
print(f"Total de peleas terminadas rápidamente (<= 05:00): {total_peleas_rapidas}")
print(f"Peleas terminadas por KO/TKO en menos de 05:00: {ko_tko_peleas_rapidas_count}")
print(f"Porcentaje de peleas terminadas por KO/TKO en peleas rápidas: {ko_tko_percentage:.2f}%")

# Verificar si el porcentaje es mayor o igual al 70% para confirmar la hipótesis
if ko_tko_percentage >= 70:
    print("La hipótesis se confirma: El 70% de las peleas rápidas son por KO/TKO.")
else:
    print("La hipótesis no se confirma: El porcentaje de peleas rápidas por KO/TKO es menor al 70%.")

# Contar el total de peleas por cada método de victoria en peleas rápidas
victorias_por_metodo = df_peleas_rapidas['Fight Method'].value_counts()

# Crear la gráfica de barras
plt.figure(figsize=(8, 6))
victorias_por_metodo.plot(kind='bar', color='skyblue')

# Personalizar el gráfico
plt.title('Métodos de Victoria en Peleas Rápidas (menos de 5 minutos)', fontsize=14)
plt.xlabel('Método de Victoria', fontsize=12)
plt.ylabel('Número de Peleas', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

########

# Sumar los golpes conectados en los primeros dos rounds para cada peleador
df['Total_Strikes_F1_R1_R2'] = df['Total Strike Landed F1R1'] + df['Total Strike Landed F1R2']
df['Total_Strikes_F2_R1_R2'] = df['Total Strike Landed F2R1'] + df['Total Strike Landed F2R2']

# Clasificar el resultado de la pelea: 1 = Peleador 1 gana, 0 = Peleador 2 gana
# Asumimos que 'Winner?' indica al ganador (1 = Peleador 1, 0 = Peleador 2)
df['F1_Wins'] = df['Winner?'] == 1

# Comparar los golpes conectados por cada peleador con el resultado de la pelea
# Evaluamos si el peleador con más golpes en los primeros dos rounds ganó la pelea
df['F1_Higher_Strikes'] = df['Total_Strikes_F1_R1_R2'] > df['Total_Strikes_F2_R1_R2']

# Ver cuántas veces el peleador con más golpes en los primeros dos rounds ganó
correct_predictions = df['F1_Higher_Strikes'] == df['F1_Wins']

# Calcular el porcentaje de veces que el peleador con más golpes en los primeros dos rounds ganó
accuracy = correct_predictions.sum() / len(df) * 100

# Mostrar el resultado
print(f'El porcentaje de veces que el peleador con más golpes en los primeros dos rounds ganó es: {accuracy:.2f}%')

# Mostrar los primeros registros para verificar si el cálculo es correcto
print(df[['Fighter1', 'Fighter2', 'Total_Strikes_F1_R1_R2', 'Total_Strikes_F2_R1_R2', 'Winner?', 'F1_Wins', 'F1_Higher_Strikes']].head())

plt.figure(figsize=(10, 6))

# Gráfico de dispersión con los golpes conectados en los primeros dos rounds en el eje X
# y el resultado de la pelea (1 = victoria del peleador 1, 0 = victoria del peleador 2) en el eje Y
sns.scatterplot(x='Total_Strikes_F1_R1_R2', y='Total_Strikes_F2_R1_R2', hue='F1_Wins', data=df, palette='coolwarm', alpha=0.6)

# Personalizar la gráfica
plt.title('Comparación de Golpes Conectados en los Primeros 2 Rounds con Resultado de la Pelea')
plt.xlabel('Golpes Conectados por Peleador 1 en Rounds 1 y 2')
plt.ylabel('Golpes Conectados por Peleador 2 en Rounds 1 y 2')
plt.legend(title='Victoria del Peleador 1', loc='upper left')

# Mostrar la gráfica
plt.show()

#######
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
# df = pd.read_csv("ruta_a_tu_archivo.csv")  # Descomenta esta línea y coloca la ruta correcta

# Hipótesis: El peleador con más derribos completados en el primer round tiene más probabilidades de ganar

# Filtrar las columnas de derribos completados en el primer round
df['Derribo_F1_R1'] = df['TD Completed F1R1']  # Derribos completados por Peleador 1 en el primer round
df['Derribo_F2_R1'] = df['TD Completed F2R1']  # Derribos completados por Peleador 2 en el primer round

# Crear una nueva columna que nos diga qué peleador completó más derribos en el primer round
df['Derribo_Mayor'] = df.apply(lambda row: 'F1' if row['Derribo_F1_R1'] > row['Derribo_F2_R1'] else 'F2', axis=1)

# Verificamos si el peleador con más derribos completados ganó
df['Resultado_Comprobacion'] = df['Derribo_Mayor'] == df['F1_Wins'].apply(lambda x: 'F1' if x == 1 else 'F2')

# Calcular el porcentaje de aciertos para la hipótesis
porcentaje_aciertos = df['Resultado_Comprobacion'].mean() * 100
print(f"Porcentaje de aciertos para la hipótesis: {porcentaje_aciertos:.2f}%")

# Crear gráfico de dispersión
plt.figure(figsize=(10, 6))

# Gráfico de dispersión con los derribos completados en el primer round de ambos peleadores
sns.scatterplot(x='Derribo_F1_R1', y='Derribo_F2_R1', hue='F1_Wins', data=df, palette='coolwarm', alpha=0.6)

# Personalizar
plt.title('Derribos Completados en el Primer Round vs Resultado de la Pelea')
plt.xlabel('Derribos Completados Peleador 1')
plt.ylabel('Derribos Completados Peleador 2')
plt.legend(title='Resultado de la Pelea', loc='upper left')

# Mostrar el gráfico
plt.show()


