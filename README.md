# Sistema de Recuperación Semántica Local (RAG)

Pipeline RAG (Retrieval-Augmented Generation) end-to-end: recibe una consulta del usuario, busca los fragmentos más relevantes en una base vectorial local (ChromaDB) y genera una respuesta que usa **exclusivamente** esa información. Si la respuesta no está en los documentos, el sistema lo dice explícitamente en vez de inventar.

## Ejemplo de uso

**Pregunta respondible** (está en `data/politica_backups.txt`):

```json
{
  "respuesta": "Los backups completos automáticos se realizan cada 24 horas y se retienen durante 30 días en almacenamiento separado de la infraestructura principal.",
  "fuentes": [
    "data/guia_onboarding_dev.txt",
    "data/politica_backups.txt",
    "data/politica_code_review.txt",
    "data/politica_incidentes.txt"
  ],
  "fragmentos_recuperados": 4
}
```

**Pregunta trampa** (no está en ningún documento):

```json
{
  "respuesta": "No tengo acceso a esa información en los documentos disponibles.",
  "fuentes": [
    "data/guia_onboarding_dev.txt",
    "data/politica_backups.txt",
    "data/politica_code_review.txt",
    "data/politica_incidentes.txt"
  ],
  "fragmentos_recuperados": 4
}
```

El campo `fuentes` lista los archivos de los fragmentos que el retriever le pasó al modelo como contexto, independientemente de si la respuesta final los usó. Como el dataset tiene 4 documentos chicos y `top_k=4`, el modelo recibe los 4 en ambos casos. Lo importante es que, ante la pregunta trampa, aun teniendo todo el contexto disponible, el modelo responde que no tiene la información en vez de inventarla. (En Windows las rutas aparecen con `\` en lugar de `/`.)

## Estructura del repositorio

```
rag-local/
├── data/                          # Dataset de ejemplo (.txt)
│   ├── politica_code_review.txt
│   ├── politica_incidentes.txt
│   ├── politica_backups.txt
│   └── guia_onboarding_dev.txt
├── schemas.py                     # Modelos Pydantic: RespuestaLLM, RAGResponse
├── ingest.py                      # Módulo de ingesta: carga, chunking y persistencia en ChromaDB
├── rag.py                         # Retriever + prompt + cadena LCEL + get_rag_response()
├── main.py                        # Mini-script de prueba asíncrono
├── requirements.txt
├── .env.example
└── .gitignore
```

El dataset es sobre políticas internas de una empresa de software ficticia ("Vertex Software"): code review, gestión de incidentes, backups y onboarding de desarrolladores. Es un tema distinto al del material de referencia de la cátedra, pero cumple la misma función: una pregunta con respuesta clara en el contexto, y una pregunta trampa sin respuesta en ningún documento.

## Cómo funciona

| Componente pedido | Dónde está | Qué hace |
|---|---|---|
| Módulo de Ingesta | `ingest.py` → `ingestar()` | Carga `.txt`/`.md` de `/data`, los fragmenta y los persiste en ChromaDB |
| Capa de Recuperación | `rag.py` → `retriever` | Convierte la pregunta en embedding y recupera los fragmentos más relevantes |
| Generación Grounded | `rag.py` → `chain` | `prompt \| llm \| PydanticOutputParser`, instruida a decir que no sabe si la respuesta no está en el contexto |
| Función asíncrona | `rag.py` → `get_rag_response()` | Orquesta los tres pasos anteriores con `.ainvoke()` |

### 1. Ingesta (`ingest.py`)

- Carga todos los `.txt` y `.md` de `/data` con `DirectoryLoader`.
- **Chunking en tokens, no en caracteres**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=70)`. El límite de contexto de un LLM se mide en tokens, así que fragmentar por cantidad de caracteres da un tamaño de chunk poco confiable. Cumple el mínimo pedido (500 tokens, 50 de overlap). Como los documentos de ejemplo son cortos (menos de 500 tokens cada uno), cada archivo queda en un solo fragmento: `chunk_size` es un techo, no un tamaño forzado. Con documentos más largos, el splitter los divide en varios fragmentos con 70 tokens de solapamiento entre ellos.
- Persiste los fragmentos en una colección de **ChromaDB** local (`./vectorstore`), con `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) — corre en CPU, no necesita API key.
- **Chequeo anti-reindexado**: si `./vectorstore` ya existe y tiene contenido, `ingestar()` carga el índice existente en vez de volver a indexar. Correr `python ingest.py` una segunda vez imprime `♻️ Índice existente — se reutiliza sin reindexar`.

### 2. Retriever (`rag.py`)

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

`top_k=4`, dentro del rango 3-5 que pide la consigna — pasar demasiados fragmentos satura el contexto del LLM y degrada la atención del modelo ("Lost in the Middle"). **Mismo modelo de embeddings para indexar y para consultar** (`ingest.get_embeddings()` se usa en los dos lugares): si no coincidieran, la distancia vectorial no tendría sentido y los resultados serían arbitrarios.

### 3. Generación grounded (`rag.py`)

El prompt de sistema instruye al modelo a responder **únicamente** con lo que está en el `CONTEXTO`, y a decir explícitamente `"No tengo acceso a esa información en los documentos disponibles."` si no lo está. La cadena LCEL es:

```python
chain = prompt | llm | parser_llm   # parser_llm: PydanticOutputParser(RespuestaLLM)
```

El LLM solo genera el texto de la respuesta (`RespuestaLLM`). Las **fuentes no se le piden al LLM** — se arman en `get_rag_response()` a partir de los metadatos reales (`source`) de los fragmentos que efectivamente recuperó el retriever. Pedirle al modelo que "recuerde" de dónde sacó la información es la forma más común de que alucine referencias inexistentes; acá las fuentes son siempre verificables porque las arma el código, no el modelo.

### 4. Función asíncrona (`rag.py`)

```python
async def get_rag_response(query: str) -> RAGResponse:
    docs = await retriever.ainvoke(query)                 # a. búsqueda de similitud
    contexto = _formatear_contexto(docs)                  # b. prompt con los fragmentos
    salida_llm = await chain.ainvoke({...})                # c. llamada asíncrona al LLM
    return RAGResponse(..., fuentes=..., ...)              # d. parseo a Pydantic + referencias
```

## Cómo ejecutarlo

Requiere Python 3.12 y un entorno virtual.

**PowerShell (Windows):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno virtual, corré una sola vez:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Git Bash / Linux / macOS:**
```bash
python -m venv .venv
source .venv/Scripts/activate    # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Después, en cualquiera de los dos casos:**
```bash
cp .env.example .env
# completar GOOGLE_API_KEY en .env (gratis en aistudio.google.com/apikey)

python main.py
```

`main.py` indexa `/data` si todavía no existe `./vectorstore`, y corre las dos pruebas pedidas: una pregunta respondible y una pregunta trampa.

Para indexar sin correr las pruebas (por ejemplo, para preparar el índice antes de desplegar):
```bash
python ingest.py
```

### Usarlo desde código

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from rag import get_rag_response

resultado = asyncio.run(get_rag_response("¿Cuántas aprobaciones necesita un PR que modifica código de facturación?"))
print(resultado.respuesta)
```

## Variables de entorno

| Variable | Requerida para |
|---|---|
| `GOOGLE_API_KEY` | Llamar al modelo Gemini (gratis en aistudio.google.com/apikey) |

El `.env` está en `.gitignore`: nunca se sube al repositorio.

## Nota sobre `sentence-transformers` en Windows con Smart App Control

Los embeddings locales (`HuggingFaceEmbeddings`) dependen de `torch`, que en algunas instalaciones de Windows 11 con **Smart App Control** activado puede fallar al importarse con un error del estilo `DLL load failed... Una directiva de Control de aplicaciones bloqueó este archivo`. No es un bug del código: es una política de seguridad del sistema operativo que bloquea binarios nativos sin reputación reconocida. Si te pasa esto, las opciones son:

- Correr el proyecto en **WSL** (Linux dentro de Windows), en Google Colab, o en Linux/macOS: Smart App Control no aplica ahí.
- Desactivar Smart App Control en **Seguridad de Windows → Control de aplicaciones y del explorador → Configuración de Smart App Control**. No permite excepciones para un archivo puntual, y según la versión de Windows puede no ser posible volver a activarlo sin reinstalar el sistema: leé el aviso de esa pantalla antes de confirmar.

## Errores comunes evitados

- **Contexto infinito**: `top_k=4` (entre 3 y 5), no se pasan decenas de fragmentos al LLM.
- **Embeddings no coincidentes**: un único punto (`ingest.get_embeddings()`) define el modelo de embeddings, usado tanto al indexar como al consultar.
- **Falta de persistencia**: `ingestar()` verifica si `./vectorstore` ya tiene datos antes de volver a indexar.
