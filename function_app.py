import json
import logging
import os
import uuid
import anthropic
import requests
import azure.functions as func

from azure.cosmos import CosmosClient, PartitionKey, exceptions

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

TONOS_VALIDOS = {"formal", "casual", "tecnico"}
_container = None


def get_container():
    """Crea el cliente una sola vez y lo reutiliza entre invocaciones."""
    global _container
    if _container is None:
        client = CosmosClient(
            os.environ["COSMOS_ENDPOINT"],
            credential=os.environ["COSMOS_KEY"],
            connection_verify=False,  # solo para el emulador (certificado autofirmado)
        )
        db = client.create_database_if_not_exists("contenido-ia")
        _container = db.create_container_if_not_exists(
            id="briefs", partition_key=PartitionKey(path="/id")
        )
    return _container


def validar_brief(data: dict) -> list[str]:
    errores = []
    tema = data.get("tema")
    if not isinstance(tema, str) or not tema.strip():
        errores.append("'tema' es obligatorio y debe ser texto")
    if data.get("tono", "formal") not in TONOS_VALIDOS:
        errores.append(f"'tono' debe ser uno de {sorted(TONOS_VALIDOS)}")
    return errores


@app.route(route="briefs", methods=["POST"])
def crear_brief(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = req.get_json()
    except ValueError:
        return _respuesta({"error": "El cuerpo debe ser JSON válido"}, 400)

    errores = validar_brief(data)
    if errores:
        return _respuesta({"errores": errores}, 400)

    brief = {
        "id": str(uuid.uuid4()),
        "tema": data["tema"].strip(),
        "tono": data.get("tono", "formal"),
        "idioma": data.get("idioma", "es"),
        "estado": "pendiente",
    }
    try:
        get_container().create_item(body=brief)
    except exceptions.CosmosHttpResponseError:
        logging.exception("No se pudo guardar el brief")
        return _respuesta({"error": "No se pudo guardar el brief"}, 503)

    logging.info("Brief guardado: %s", brief["id"])
    return _respuesta(brief, 201)


@app.route(route="briefs/{id}", methods=["GET"])
def obtener_brief(req: func.HttpRequest) -> func.HttpResponse:
    brief_id = req.route_params.get("id")
    try:
        brief = get_container().read_item(item=brief_id, partition_key=brief_id)
    except exceptions.CosmosResourceNotFoundError:
        return _respuesta({"error": "Brief no encontrado"}, 404)
    except exceptions.CosmosHttpResponseError:
        logging.exception("No se pudo leer el brief")
        return _respuesta({"error": "No se pudo leer el brief"}, 503)

    brief = {k: v for k, v in brief.items() if not k.startswith("_")}
    return _respuesta(brief, 200)


def _respuesta(cuerpo: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(cuerpo, ensure_ascii=False),
        status_code=status,
        mimetype="application/json",
        charset="utf-8",
    )

@app.function_name(name="procesar_brief")
@app.cosmos_db_trigger(
    arg_name="briefs",
    database_name="contenido-ia",
    container_name="briefs",
    connection="COSMOS_CONEXION",
    lease_container_name="leases",
    create_lease_container_if_not_exists=True,
)
def procesar_brief(briefs: func.DocumentList) -> None:
    for doc in briefs:
        brief = dict(doc)
        if brief.get("estado") != "pendiente":
            continue  # idempotencia: si ya fue procesado, lo saltamos

        USAR_IA_REAL = False  # cambia a True cuando tengas crédito en Anthropic

        if USAR_IA_REAL:
            try:
                contenido = generar_contenido(brief["tema"], brief["tono"], brief["idioma"])
            except anthropic.APIError:
                logging.error("No se generó contenido para %s, se reintentará en el próximo Change Feed", brief["id"])
                continue  # dejamos el estado en 'pendiente' para reintentar más tarde
        else:
            contenido = generar_contenido_simulado(brief["tema"], brief["tono"])

        brief["contenido"] = contenido
        brief["estado"] = "generado"
        get_container().upsert_item(body=brief)
        logging.info("Brief procesado: %s", brief["id"])
        
        publicar_en_cms(brief, contenido)




_cliente_ia = None


def get_cliente_ia() -> anthropic.Anthropic:
    global _cliente_ia
    if _cliente_ia is None:
        _cliente_ia = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _cliente_ia


def generar_contenido(tema: str, tono: str, idioma: str) -> str:
    prompt = (
        f"Escribe una descripción de producto breve (máximo 3 frases) "
        f"sobre: {tema}. Tono: {tono}. Idioma: {idioma}. "
        f"Devuelve solo el texto, sin explicaciones adicionales."
    )
    try:
        respuesta = get_cliente_ia().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return respuesta.content[0].text.strip()
    except anthropic.APIError:
        logging.exception("Fallo al llamar a la API de Anthropic")
        raise



def generar_contenido_simulado(tema: str, tono: str) -> str:
    """Usado mientras no hay crédito de API. Sustituible por generar_contenido()."""
    return f"[Borrador {tono}] Contenido generado para: {tema}."





def publicar_en_cms(brief: dict, contenido: str) -> None:
    payload = {
        "data": {
            "tema": brief["tema"],
            "tono": brief["tono"],
            "contenido": contenido,
            "brief_id": brief["id"],
        }
    }
    try:
        respuesta = requests.post(os.environ["STRAPI_URL"], json=payload, timeout=10)
        respuesta.raise_for_status()
        logging.info("Publicado en Strapi: %s", brief["id"])
    except requests.RequestException:
        logging.exception("No se pudo publicar en Strapi para el brief %s", brief["id"])
        # No relanzamos el error: el contenido ya se generó y guardó en Cosmos DB,
        # así que no queremos perder ese trabajo solo porque el CMS falló.