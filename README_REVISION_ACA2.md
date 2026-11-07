# ACA 2 – Redes neuronales y despliegue Streamlit

## Contenido
- `ACA2_Redes_Neuronales_Comparacion_Interpretaciones_v7.ipynb`: notebook actualizado, conserva las explicaciones por etapa y las gráficas existentes; incorpora interpretaciones automáticas de métricas, matriz de confusión, curvas y comparación de arquitecturas.
- `app.py`: aplicación Streamlit entregada como referencia del despliegue actual. No se modificó para evitar alterar la versión publicada.
- `ACA2_Redes_Neuronales_Comparacion_Interpretaciones_v7.py`: exportación Python del notebook.

## Flujo recomendado
1. Ejecutar el notebook de principio a fin en Colab con el dataset original.
2. Descargar el ZIP `artifacts_neural.zip` generado al final.
3. Descomprimir `artifacts_neural/` en la raíz del repositorio de GitHub, junto con `app.py`, el CSV del dataset y `requirements.txt`.
4. Verificar que `app.py` cargue los archivos `artifacts_neural/metadata.joblib`, `preprocessor.joblib`, `neural_metrics.joblib`, `neural_details.joblib` y los tres modelos `.keras`.
5. En Streamlit Community Cloud, configurar el archivo principal como `app.py` y desplegar desde la rama correcta.

## Nota metodológica
Las gráficas y las interpretaciones se generan a partir de los resultados de la ejecución. La tabla de umbrales se conserva como análisis de sensibilidad; el umbral definitivo no debe seleccionarse usando el conjunto de prueba. La comparación de ACA 1 con ACA 2 se considera descriptiva si no comparten partición de prueba.
