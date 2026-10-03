"""Mini-script de prueba asíncrono del pipeline RAG.

Corre dos preguntas contra get_rag_response(): una cuya respuesta está en los
documentos de /data, y una "pregunta trampa" cuya respuesta no está, para
comprobar que el modelo no alucina.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# Los SDKs y librerías de ML son muy verbosos a nivel INFO; los bajamos para
# que no tapen los logs propios del pipeline.
for nombre in ("httpx", "httpx2", "google_genai", "chromadb", "sentence_transformers"):
    logging.getLogger(nombre).setLevel(logging.WARNING)

PREGUNTA_RESPONDIBLE = "¿Cada cuánto se hacen los backups completos y cuánto tiempo se retienen?"
PREGUNTA_TRAMPA = "¿Cuál es la política de bonos por antigüedad en la empresa?"


async def probar(get_rag_response, titulo: str, pregunta: str) -> None:
    print(f"\n=== {titulo} ===")
    print(f"Pregunta: {pregunta}")
    resultado = await get_rag_response(pregunta)
    print("Respuesta:", resultado.respuesta)
    print("Fuentes:", ", ".join(resultado.fuentes) if resultado.fuentes else "—")
    print("Fragmentos recuperados:", resultado.fragmentos_recuperados)


async def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Falta GOOGLE_API_KEY en el entorno. Completá el archivo .env (ver .env.example).")
        return

    # Import diferido: rag.py construye el cliente de Gemini al importarse,
    # así que lo hacemos recién acá, una vez confirmado que la key está cargada.
    from rag import get_rag_response

    await probar(get_rag_response, "Pregunta respondible", PREGUNTA_RESPONDIBLE)
    await probar(get_rag_response, "Pregunta trampa (sin respuesta en el contexto)", PREGUNTA_TRAMPA)


if __name__ == "__main__":
    asyncio.run(main())
