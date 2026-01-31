from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from django.views.decorators.cache import cache_page

from openai import OpenAI
import pandas as pd
import io
import base64
from django.shortcuts import render
from django.http import JsonResponse
from datetime import datetime
from pathlib import Path
from django.conf import settings
from functools import lru_cache

CSV_PATH = os.environ.get("CSV_PATH")

if CSV_PATH:
    csv_path = Path(CSV_PATH)
else:
    csv_path = Path(settings.BASE_DIR) / "core" / "static" / "core" / "bbdd_full.csv"

import pandas as pd

import io, base64
from datetime import datetime, timedelta
from django.shortcuts import render

# intenta usar la librería 'holidays' si está instalada
try:
    import holidays as pyholidays
except Exception:
    pyholidays = None

# constantes / mappings (asegúrate de tener INTERVALOS y ZONAS definidos)
INTERVALOS = {
    "h": "h",
    "hora": "h",
    "d": "D",
    "dia": "D",
    "w": "W",
    "semana": "W",
    "m": "M",
    "mes": "M",
}

ZONAS = {
    # canonical -> csv column
    "total": "energia_total_kwh",

    "comedor": "energia_comedor_kwh",
    "comedores": "energia_comedor_kwh",

    "salon": "energia_salones_kwh",
    "salones": "energia_salones_kwh",

    "laboratorio": "energia_laboratorios_kwh",
    "laboratorios": "energia_laboratorios_kwh",

    "auditorio": "energia_auditorios_kwh",
    "auditorios": "energia_auditorios_kwh",

    "oficina": "energia_oficinas_kwh",
    "oficinas": "energia_oficinas_kwh",
}


# multiplicadores para thresholds (ajústalos si quieres)
TH_WARNING = 1.5
TH_ALERT = 2.0
_DF_CACHE = None

def load_csv():
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = pd.read_csv(
            "core/static/core/bbdd_full.csv", sep=";"
        )
    return _DF_CACHE

@cache_page(60 * 10)
def consumo_energia(request):
    # =========================
    # IMPORTS SEGUROS (HEADLESS)
    # =========================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # -------------------------
    # Params
    # -------------------------
    sede = request.GET.get("sede", "Tunja")
    zona = request.GET.get("zona", "total")
    modo = request.GET.get("modo", "auto")
    intervalo_req = request.GET.get("intervalo", "h")
    fecha = request.GET.get("fecha")
    inicio = request.GET.get("inicio")
    fin = request.GET.get("fin")

    freq = INTERVALOS.get(intervalo_req, "H")

    # -------------------------
    # Leer CSV (cacheado)
    # -------------------------
    try:
        df = load_csv().copy()
    except Exception as e:
        return render(request, "core/energia.html", {
            "error": f"Error leyendo CSV: {e}"
        })

    # -------------------------
    # Limpieza básica
    # -------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])

    if "sede" not in df.columns:
        return render(request, "core/energia.html", {
            "error": "CSV no contiene columna 'sede'."
        })

    df = df[df["sede"] == sede]
    if df.empty:
        return render(request, "core/energia.html", {
            "error": f"No hay datos para la sede {sede}."
        })

    min_fecha = df["timestamp"].min()
    max_fecha = df["timestamp"].max()

    if zona not in ZONAS or ZONAS[zona] not in df.columns:
        return render(request, "core/energia.html", {
            "error": f"Zona inválida: {zona}"
        })

    columna = ZONAS[zona]

    # normalizar números latinos
    df[columna] = (
        df[columna]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.dropna(subset=[columna])

    if df.empty:
        return render(request, "core/energia.html", {
            "error": "No hay datos numéricos válidos."
        })

    # -------------------------
    # Rango temporal
    # -------------------------
    if modo == "auto":
        inicio_dt, fin_dt = min_fecha, max_fecha

    elif modo == "24h":
        fin_dt = max_fecha
        inicio_dt = fin_dt - timedelta(hours=24)

    elif modo == "dia":
        if not fecha:
            return render(request, "core/energia.html", {
                "error": "Falta fecha."
            })
        dia = pd.to_datetime(fecha, errors="coerce")
        if pd.isna(dia):
            return render(request, "core/energia.html", {
                "error": "Fecha inválida."
            })
        inicio_dt = dia.normalize()
        fin_dt = inicio_dt + timedelta(days=1)

    elif modo == "rango":
        inicio_dt = pd.to_datetime(inicio, errors="coerce")
        fin_dt = pd.to_datetime(fin, errors="coerce")
        if pd.isna(inicio_dt) or pd.isna(fin_dt):
            return render(request, "core/energia.html", {
                "error": "Rango inválido."
            })

    else:
        return render(request, "core/energia.html", {
            "error": "Modo inválido."
        })

    inicio_dt = max(inicio_dt, min_fecha)
    fin_dt = min(fin_dt, max_fecha)

    df = df[(df["timestamp"] >= inicio_dt) & (df["timestamp"] <= fin_dt)]
    if df.empty:
        return render(request, "core/energia.html", {
            "error": "No hay datos en el rango."
        })

    # -------------------------
    # Serie final
    # -------------------------
    df = df.set_index("timestamp").sort_index()
    serie = df[columna].resample(freq).sum()

    full_index = pd.date_range(start=inicio_dt, end=fin_dt, freq=freq)
    serie = serie.reindex(full_index, fill_value=0)

    # 🔒 límite duro de puntos (CRÍTICO EN PROD)
    MAX_POINTS = 500
    if len(serie) > MAX_POINTS:
        step = max(1, len(serie) // MAX_POINTS)
        serie_plot = serie.iloc[::step]
    else:
        serie_plot = serie

    # -------------------------
    # Plot SEGURO
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        serie_plot.index,
        serie_plot.values,
        linewidth=1,
        label="Consumo"
    )

    locator = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    ax.set_ylabel("kWh")
    ax.set_title(f"{sede} · {zona} · {modo}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # -------------------------
    # Export imagen
    # -------------------------
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )
    plt.close(fig)

    buf.seek(0)
    graphic = base64.b64encode(buf.read()).decode()

    # -------------------------
    # Context
    # -------------------------
    context = {
        "grafico": graphic,
        "sede": sede,
        "zona": zona,
        "modo": modo,
        "intervalo": freq,
        "min_fecha": min_fecha.isoformat(),
        "max_fecha": max_fecha.isoformat(),
    }

    return render(request, "core/energia.html", context)


# =========================
# Configuración OpenAI
# =========================
client = OpenAI(
    api_key=os.environ.get("API", "sk-")
)

SYSTEM_PROMPT = """
Eres un auditor energético experto en la norma ISO 50001.

REGLAS ESTRICTAS:
- No inventes datos.
- No hagas cálculos numéricos.
- Usa únicamente la información proporcionada en el contexto.
- Si falta sede o zona para responder una pregunta de consumo, debes pedir que se especifiquen.
- Cuando se solicite un “informe de auditoría”, responde en formato estructurado con:
  1. Métodos
  2. Métricas
  3. Hallazgos
  4. Anomalías
  5. Recomendaciones
  6. Conclusión
- Redacta con lenguaje técnico acorde a ISO 50001.

"""


# =========================
# Vistas de páginas
# =========================
def inicio(request):
    return render(request, 'core/inicio.html')


def uso_ia(request):
    return render(request, 'core/uso_ia.html')


def chatbot_page(request):
    return render(request, 'core/chatbot.html')


# =========================
# API del Chatbot
# =========================
@csrf_exempt
def chat_api(request):
    """
    Endpoint POST
    Recibe JSON:
      {
        "message": "texto del usuario",
        "sede": "Tunja",            # opcional (recomendado para auditoría/consultas numéricas)
        "zona": "total",            # opcional (recomendado)
        "inicio": "2025-01-01T00:00:00",  # opcional
        "fin": "2025-01-31T23:59:59",     # opcional
        "csv_base64": "...",        # opcional: base64 del CSV si quieres subir CSV en la petición
      }
    Devuelve: { "reply": "respuesta del bot" }
    ------------------------------------------------------------
    Notas:
    - Esta versión extrae 'sede' y 'zona' desde lenguaje natural (ej: "en tunja los laboratorios").
    - Intenta cargar el CSV desde payload (csv_base64) o desde varias rutas del servidor.
    - No se envía el CSV crudo a OpenAI: se envía un contexto JSON procesado.
    - Incluye un PRE_PROMPT con sedes, zonas y columnas del CSV (útil para system prompts).
    """
    import os
    import re
    import json
    import base64
    import traceback
    from io import BytesIO

    # Dependencias que se usan internamente (pd debe estar disponible globalmente
    # en tu archivo; si no, importa aquí)
    try:
        import pandas as pd
    except Exception:
        return JsonResponse({"reply": "Error: pandas no disponible en el entorno."}, status=500)

    # ----------------------------------------
    # Config: sedes, zonas y columnas conocidas
    # ----------------------------------------
    # canonical names (cómo queremos devolverlas)
    SEDES_CANONICAL = {
        "tunja": "Tunja",
        "duitama": "Duitama",
        "sogamoso": "Sogamoso",
        "chiquinquira": "Chiquinquirá",
        "chiquinquirá": "Chiquinquirá",  # aceptar ambas
        "chiquinquera": "Chiquinquirá",  # por si hay variantes
    }

    # ZONAS -> columna del CSV (asegúrate de mantener en sync con tu CSV)
    ZONAS = {
    # canonical -> columna CSV
    "total": "energia_total_kwh",

    "comedor": "energia_comedor_kwh",
    "comedores": "energia_comedor_kwh",

    "salon": "energia_salones_kwh",
    "salones": "energia_salones_kwh",

    "laboratorio": "energia_laboratorios_kwh",
    "laboratorios": "energia_laboratorios_kwh",

    "auditorio": "energia_auditorios_kwh",
    "auditorios": "energia_auditorios_kwh",

    "oficina": "energia_oficinas_kwh",
    "oficinas": "energia_oficinas_kwh",
}


    CSV_COLUMNS_EXAMPLE = [
        "reading_id", "timestamp", "sede", "sede_id", "energia_total_kwh",
        "energia_comedor_kwh", "energia_salones_kwh", "energia_laboratorios_kwh",
        "energia_auditorios_kwh", "energia_oficinas_kwh", "potencia_total_kw",
        "agua_litros", "temperatura_exterior_c", "ocupacion_pct",
        "hora", "dia_semana", "dia_nombre", "mes", "trimestre", "año",
        "periodo_academico", "es_fin_semana", "es_festivo", "es_semana_parciales",
        "es_semana_finales", "co2_kg"
    ]

    # Pre-prompt (útil para pasarlo al modelo en llamadas tipo system)
    PRE_PROMPT = (
        "Contexto: las sedes disponibles son: Tunja, Duitama, Sogamoso, Chiquinquirá. "
        "Las zonas disponibles son: comedor, salones, laboratorios, auditorios, oficinas, total.\n"
        f"Columnas relevantes del CSV: {', '.join(CSV_COLUMNS_EXAMPLE)}.\n"
        "Regla de extracción: cuando un usuario pregunte por promedios o generación de auditoría, "
        "necesitamos 'sede' y 'zona' explícitas. Extrae la sede y la zona del texto del usuario si aparecen."
        "El valor por kwh es de 1200 COP, siempre que tengas valores de KWH calcula el total y mencionalo"
    )

    # ----------------------------------------
    # Intent detection keywords
    # ----------------------------------------
    AUDIT_KEYWORDS = [
        "auditor", "auditoría", "auditoria", "informe de auditoría",
        "informe auditoría", "iso 50001", "iso50001", "generar informe",
        "generame un informe", "generar auditoría", "generar informe iso"
    ]
    AVG_KEYWORDS = [
        "promedio", "consumo promedio", "media de consumo", "average", "promedio de consumo",
        "consumo medio", "cuál es el promedio", "cual es el promedio"
    ]

    # ----------------------------------------
    # CSV candidate paths (prioridad)
    # ----------------------------------------
    CSV_CANDIDATES = []

    # 1. Desde variable de entorno (PRIORIDAD)
    csv_env = os.environ.get("CSV_PATH")
    if csv_env:
        CSV_CANDIDATES.append(Path(settings.BASE_DIR) / csv_env)

    # 2. Fallback controlado (opcional, pero válido)
    CSV_CANDIDATES.append(
        Path(settings.BASE_DIR) / "core" / "data" / "bbdd_full.csv"
    )

    # Filtrar solo los que existen
    CSV_CANDIDATES = [p for p in CSV_CANDIDATES if p.exists()]

    # ----------------------------------------
    # Helpers
    # ----------------------------------------
    def _normalize_text(s: str) -> str:
        """Lowercase, strip, remove diacritics simples and unify whitespace/punctuation."""
        s = s.lower()
        # replace common accents manually (no dependencia externa)
        accent_map = str.maketrans(
            "áéíóúüñÁÉÍÓÚÜÑ",
            "aeiouunAEIOUUN"
        )
        s = s.translate(accent_map)
        # replace punctuation with spaces
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _try_read_csv_bytes(raw_bytes):
        """Intentar leer CSV con diferentes separadores y codificaciones."""
        for sep in [";", ",", "\t"]:
            try:
                return pd.read_csv(BytesIO(raw_bytes), sep=sep, engine="python", encoding="utf-8")
            except Exception:
                try:
                    return pd.read_csv(BytesIO(raw_bytes), sep=sep, engine="python", encoding="latin-1")
                except Exception:
                    continue
        return None
    def build_llm_safe_context(ctx, max_anomalies=20):
        """
        Reduce el contexto para enviarlo a la IA:
        - Quita listas grandes
        - Limita anomalías
        - Mantiene métricas clave
        """
        if not ctx or "metricas" not in ctx:
            return ctx

        resumen = {
            "sede": ctx.get("sede"),
            "zona": ctx.get("zona"),
            "periodo": {
                "inicio": ctx.get("periodo", {}).get("applied_inicio") or ctx.get("min_fecha"),
                "fin": ctx.get("periodo", {}).get("applied_fin") or ctx.get("max_fecha"),
            },
            "metricas": ctx.get("metricas"),
            "total_anomalias": ctx.get("total_anomalias"),
            "anomalias_muestra": ctx.get("anomalias", [])[:max_anomalies],
            "nota_anomalias": (
                "Se muestra solo una muestra representativa de anomalías. "
                "El total completo fue analizado en backend."
            )
        }

        return resumen
    def load_csv_from_payload_or_disk(payload):
        """Carga CSV desde payload (base64) o desde disco (varias rutas)."""
        # 1) csv_base64 in payload
        csv_b64 = payload.get("csv_base64") if isinstance(payload, dict) else None
        if csv_b64:
            try:
                raw = base64.b64decode(csv_b64)
                df_try = _try_read_csv_bytes(raw)
                if df_try is not None:
                    return df_try
            except Exception:
                pass

        # 2) intentar rutas del servidor
        for path in CSV_CANDIDATES:
            try:
                if path and os.path.exists(path):
                    # intentar con varios separadores automáticamente
                    with open(path, "rb") as f:
                        raw = f.read()
                    df_try = _try_read_csv_bytes(raw)
                    if df_try is not None:
                        return df_try
            except Exception:
                continue

        # 3) not found
        return None

    def extract_sede_zona_from_text(text):
        if not text:
            return None, None

        norm = _normalize_text(text)

        detected_sede = None
        detected_zona = None

        # detectar sede
        for key_norm, canonical in SEDES_CANONICAL.items():
            if key_norm in norm:
                detected_sede = canonical
                break

        # detectar zona (plural/singular)
        for zkey in ZONAS.keys():
            if zkey in norm:
                detected_zona = zkey
                break

        return detected_sede, detected_zona


    # ----------------------------------------
    # build_audit_context (misma lógica que ya tenías pero encapsulada)
    # ----------------------------------------
    def build_audit_context(df, sede, zona, inicio=None, fin=None):
        result = {
            "sede": sede,
            "zona": zona,
            "periodo": {"requested_inicio": inicio, "requested_fin": fin},
            "min_fecha": None,
            "max_fecha": None,
            "metricas": {},
            "umbrales": {},
            "anomalias": [],
            "total_anomalias": 0
        }

        if df is None:
            return {"error": "CSV no disponible."}

        if "timestamp" not in df.columns:
            return {"error": "CSV no contiene columna 'timestamp'."}
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        if "sede" in df.columns and sede:
            # comparar normalizando
            df["sede_norm"] = df["sede"].astype(str).str.lower().str.normalize("NFKD")
            # but normalization above may not remove accents consistently; do simple compare
            df = df[df["sede"].astype(str).str.lower().str.contains(sede.lower())]
        if df.empty:
            return {"error": f"No hay datos para la sede {sede}."}

        min_fecha = df["timestamp"].min()
        max_fecha = df["timestamp"].max()
        result["min_fecha"] = min_fecha.isoformat()
        result["max_fecha"] = max_fecha.isoformat()

        if zona not in ZONAS:
            return {"error": f"Zona inválida: {zona}"}
        columna = ZONAS[zona]
        if columna not in df.columns:
            return {"error": f"Columna {columna} no encontrada en el CSV."}

        # normalizar números latinos comunes
        df[columna] = (
            df[columna].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df = df.dropna(subset=[columna])
        if df.empty:
            return {"error": "No hay datos numéricos en la columna seleccionada."}

        # aplicar rango si viene
        try:
            if inicio:
                inicio_dt = pd.to_datetime(inicio, errors="coerce")
                if not pd.isna(inicio_dt):
                    df = df[df["timestamp"] >= inicio_dt]
                    result["periodo"]["applied_inicio"] = inicio_dt.isoformat()
        except Exception:
            pass
        try:
            if fin:
                fin_dt = pd.to_datetime(fin, errors="coerce")
                if not pd.isna(fin_dt):
                    df = df[df["timestamp"] <= fin_dt]
                    result["periodo"]["applied_fin"] = fin_dt.isoformat()
        except Exception:
            pass

        if df.empty:
            return {"error": "No hay datos en el periodo solicitado."}

        # resample horario base
        df = df.set_index("timestamp").sort_index()
        hourly = df[columna].resample("h").sum().to_frame(name="consumo")
        if hourly.empty:
            return {"error": "No hay datos horarios para el análisis."}

        # detectar días festivos básico (sin librería)
        holidays_set = set()
        try:
            if "pyholidays" in globals() and globals().get("pyholidays") is not None:
                hols = globals()["pyholidays"].CountryHoliday("CO", years=range(min_fecha.year, max_fecha.year + 1))
                holidays_set = set(hols.keys())
        except Exception:
            holidays_set = set()

        def section_from_hour(h):
            if 0 <= h < 6:
                return "00-06"
            if 6 <= h < 12:
                return "06-12"
            if 12 <= h < 18:
                return "12-18"
            return "18-24"

        hourly = hourly.reset_index()
        hourly["date"] = hourly["timestamp"].dt.date
        hourly["hour"] = hourly["timestamp"].dt.hour
        hourly["weekday"] = hourly["timestamp"].dt.weekday
        hourly["day_type"] = hourly["date"].apply(
            lambda d: "holiday" if d in holidays_set else ("weekend" if pd.Timestamp(d).weekday() >= 5 else "weekday")
        )
        hourly["section"] = hourly["hour"].apply(section_from_hour)

        stats = hourly.groupby(["day_type", "section"])["consumo"].mean().unstack(fill_value=0)

        promedio = {}
        umbrales = {}
        TH_WARNING = 1.5
        TH_ALERT = 2.0
        for dt in ["weekday", "weekend", "holiday"]:
            promedio[dt] = {}
            umbrales[dt] = {}
            for sec in ["00-06", "06-12", "12-18", "18-24"]:
                mean_val = float(stats.loc[dt, sec]) if (dt in stats.index and sec in stats.columns) else 0.0
                promedio[dt][sec] = mean_val
                umbrales[dt][sec] = {
                    "warning": mean_val * TH_WARNING if mean_val > 0 else None,
                    "alert": mean_val * TH_ALERT if mean_val > 0 else None
                }

        serie = df[columna].resample("h").sum()
        try:
            full_index = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq="h")
            serie = serie.reindex(full_index, fill_value=0)
        except Exception:
            pass

        alerts = []
        for ts, val in serie.items():
            d = ts.date()
            h = ts.hour
            sec = section_from_hour(h)
            if d in holidays_set:
                dt_type = "holiday"
            elif ts.weekday() >= 5:
                dt_type = "weekend"
            else:
                dt_type = "weekday"

            th = umbrales.get(dt_type, {}).get(sec, {})
            alert_thr = th.get("alert") if th else None
            warn_thr = th.get("warning") if th else None

            level = None
            if alert_thr and val > alert_thr:
                level = "ALERTA"
            elif warn_thr and val > warn_thr:
                level = "WARNING"

            if level:
                alerts.append({
                    "timestamp": ts.isoformat(),
                    "value": float(val),
                    "level": level,
                    "threshold": alert_thr if level == "ALERTA" else warn_thr,
                    "day_type": dt_type,
                    "section": sec
                })

        result["metricas"] = {
            "consumo_total_kwh": float(df[columna].sum()),
            "consumo_promedio_hora": float(df[columna].mean()),
            "promedios_por_seccion": promedio
        }
        result["umbrales"] = umbrales
        result["anomalias"] = alerts
        result["total_anomalias"] = len(alerts)

        return result

    # ----------------------------------------
    # Inicio del endpoint
    # ----------------------------------------
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
    except Exception:
        return JsonResponse({"reply": "Error leyendo el body JSON."}, status=400)

    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return JsonResponse({"reply": "No recibí ningún mensaje."})

    lower_msg = user_message.lower()
    is_audit_request = any(k in lower_msg for k in AUDIT_KEYWORDS)
    is_avg_request = any(k in lower_msg for k in AVG_KEYWORDS)

    # parámetros opcionales del payload (prioritarios frente a extracción por texto)
    sede = payload.get("sede") or payload.get("site") or None
    zona = payload.get("zona") or payload.get("zone") or None
    inicio = payload.get("inicio")
    fin = payload.get("fin")

    # Si no se pasan por payload, intentar extraer desde el texto
    if not sede or not zona:
        detected_sede, detected_zona = extract_sede_zona_from_text(user_message)
        if not sede and detected_sede:
            sede = detected_sede
        if not zona and detected_zona:
            zona = detected_zona

    # cargar CSV
    df = load_csv_from_payload_or_disk(payload)

    # -------------------------
    # Reglas: promedio
    # -------------------------
    if is_avg_request:
        if not sede or not zona:
            return JsonResponse({
                "reply": "Para calcular el promedio de consumo necesito que especifiques la sede y la zona (por ejemplo: 'sede=Tunja' y 'zona=salones')."
            })
        ctx = build_audit_context(df, sede, zona, inicio, fin)
        if "error" in ctx:
            return JsonResponse({"reply": f"Error al procesar datos: {ctx['error']}"} )
        promedio_hora = ctx["metricas"].get("consumo_promedio_hora")
        consumo_total = ctx["metricas"].get("consumo_total_kwh")
        reply_text = (
            f"Promedio horario para sede='{sede}', zona='{zona}' "
            f"entre {ctx['periodo'].get('applied_inicio', ctx['min_fecha'])} y {ctx['periodo'].get('applied_fin', ctx['max_fecha'])}:\n"
            f"- Consumo promedio por hora: {promedio_hora:.3f} kWh\n"
            f"- Consumo total en el periodo: {consumo_total:.3f} kWh\n"
            f"- Total anomalías detectadas: {ctx['total_anomalias']}"
        )
        return JsonResponse({"reply": reply_text})

    # -------------------------
    # Reglas: auditoría ISO 50001
    # -------------------------
    if is_audit_request:
        if not sede or not zona:
            return JsonResponse({"reply": "Para generar un informe de auditoría necesito que especifiques 'sede' y 'zona' en la petición (o escríbelas en el mensaje)."})
        ctx = build_audit_context(df, sede, zona, inicio, fin)
        if "error" in ctx:
            return JsonResponse({"reply": f"Error al procesar los datos: {ctx['error']}"} )

        # System prompt estricto + PRE_PROMPT
        SYSTEM_AUDIT_PROMPT = PRE_PROMPT + "\n\n" + (
            "Eres un auditor energético experto en la norma ISO 50001.\n"
            "REGLAS ESTRICTAS:\n"
            "- No inventes datos.\n"
            "- No hagas cálculos numéricos nuevos (usa sólo las métricas que vienen en el contexto).\n"
            "- Usa únicamente la información proporcionada en el contexto JSON.\n"
            "- Si falta información crítica, dilo explícitamente.\n"
            "- Cuando se solicite un “informe de auditoría”, responde en formato estructurado con:\n"
            "  1. Métodos (breve descripción de cómo se obtuvieron las métricas)\n"
            "  2. Métricas (resumen de los números provistos)\n"
            "  3. Hallazgos (interpretación)\n"
            "  4. Anomalías (listar y explicar)\n"
            "  5. Recomendaciones (acciones concretas)\n"
            "  6. Conclusión (resumen ejecutivo)\n"
            "- Utiliza lenguaje técnico pero claro; menciona ISO 50001 donde corresponda."
        )

        safe_ctx = build_llm_safe_context(ctx)

        user_payload_for_model = {
            "instruction": "Genera un informe de auditoría energética conforme ISO 50001 usando únicamente el contexto proporcionado.",
            "context": safe_ctx
        }


        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_AUDIT_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload_for_model, ensure_ascii=False, indent=2)}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            reply = response.choices[0].message.content
        except Exception as e:
            tb = traceback.format_exc()
            reply = f"Error al consultar la IA para generar informe: {str(e)}\n{tb}"

        return JsonResponse({"reply": reply})

    # -------------------------
    # Caso por defecto: conversación general
    # -------------------------
    SYSTEM_GENERIC_PROMPT = PRE_PROMPT + "\n\n" + (
        "Eres un asistente experto en consumo energético que debe atenerse a la información proporcionada.\n"
        "REGLAS: No inventes datos operativos; si el usuario requiere promedios o informes y no dio sede/zona, pide que los especifique."
        ""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_GENERIC_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,
            max_tokens=300
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error al consultar la IA: {str(e)}"

    return JsonResponse({"reply": reply})
