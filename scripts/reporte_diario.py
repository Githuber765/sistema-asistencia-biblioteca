from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "data" / "asistencia.csv"
OUT_DIR = ROOT / "data" / "reportes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():
    print("No hay registros de asistencia todavía.")
    raise SystemExit(0)

# Leer CSV con la columna timestamp como datetime
df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
if df.empty:
    print("El CSV está vacío.")
    raise SystemExit(0)

# Columnas auxiliares
df["fecha"] = df["timestamp"].dt.date

# Tomo la última fecha que haya en el CSV (día más reciente)
ultima_fecha = df["fecha"].max()
df_dia = df[df["fecha"] == ultima_fecha]

# Solo eventos de entrada
df_in = df_dia[df_dia["evento"] == "IN"].copy()

# Conteo por persona
reporte = (
    df_in.groupby(["fecha", "nombre"])
         .size()
         .reset_index(name="asistencias")
         .sort_values(["asistencias", "nombre"], ascending=[False, True])
)

OUT_FILE = OUT_DIR / "reporte_diario.csv"
reporte.to_csv(OUT_FILE, index=False, encoding="utf-8")

print(f" Reporte generado: {OUT_FILE}")
print(reporte if not reporte.empty else f"Sin asistencias IN en {ultima_fecha}.")
