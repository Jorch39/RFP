import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar base de conocimiento
base_df = pd.read_csv("base.csv")  # columnas: Pregunta, Respuestas
base_df = base_df.dropna(subset=["Pregunta", "Respuestas"])  # limpia filas vacías
base_preguntas = base_df["Pregunta"].tolist()
base_embeddings = model.encode(base_preguntas, convert_to_tensor=True)

# Cargar nuevas preguntas
nuevas_df = pd.read_csv("nuevas_preguntas.csv")  # columna: Pregunta
nuevas_df = nuevas_df.dropna(subset=["Pregunta"])
nuevas_preguntas = nuevas_df["Pregunta"].tolist()

# Procesar preguntas
resultados = []

for pregunta_usuario in nuevas_preguntas:
    embedding = model.encode(pregunta_usuario, convert_to_tensor=True)
    similitudes = util.cos_sim(embedding, base_embeddings)[0]
    top_scores, top_indices = similitudes.topk(3)

    for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
        resultados.append({
            "Pregunta de Entrada": pregunta_usuario,
            "Respuesta Nº": rank,
            "Respuesta Similar": base_df.iloc[idx]["Respuestas"],
            "Similitud (Probabilidad Aproximada)": round(float(score), 4)
        })

# Guardar resultados
output_df = pd.DataFrame(resultados)
output_df.to_csv("output.csv", index=False)
print("✅ Proceso completado. Resultados guardados en 'output.csv'")
