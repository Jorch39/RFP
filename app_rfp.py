import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

st.title("🔍 Buscador de Respuestas Inteligente")
st.markdown("Sube un archivo con preguntas nuevas para obtener respuestas automáticas.")

# Cargar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar base de conocimiento
base_df = pd.read_csv("Base.csv")  # columnas: Pregunta, Respuestas
base_df = base_df.dropna(subset=["Pregunta", "Respuestas"])  # limpia filas vacías
base_preguntas = base_df["Pregunta"].tolist()
base_embeddings = model.encode(base_preguntas, convert_to_tensor=True)

preguntas_file = st.file_uploader("📝 Sube las nuevas preguntas (.csv)", type=["csv"])

#if preguntas_file:
    # Leer archivos
nuevas_df = pd.read_csv(preguntas_file)

# Cargar nuevas preguntas
#nuevas_df = pd.read_csv("nuevas_preguntas.csv")  # columna: Pregunta
nuevas_df = nuevas_df.dropna(subset=["Pregunta"])
nuevas_preguntas = nuevas_df["Pregunta"].tolist()

st.spinner("Procesando preguntas...")
resultados = []

# Procesar preguntas  
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
#output_df.to_csv("output.csv", index=False)
print("✅ Proceso completado. Resultados guardados en 'output.csv'")
st.success("✅ Procesamiento completo")
st.write("📄 Resultados:")
st.dataframe(output_df)

# Descargar CSV
csv = output_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar resultados como CSV", csv, "respuestas_generadas.csv", "text/csv")
