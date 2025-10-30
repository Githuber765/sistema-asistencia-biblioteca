from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "data" / "asistencia.csv"
OUT_DIR = ROOT / "data" / "reportes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():
    print("No hay registros de asistencia todavía.")
    raise SystemExit(0)

df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
if df.empty:
    print("El CSV está vacío.")
    raise SystemExit(0)

df["anio"] = df["timestamp"].dt.year
df["mes"]  = df["timestamp"].dt.month

anio_act = df["anio"].max()
mes_act  = df[df["anio"] == anio_act]["mes"].max()

df_mes = df[(df["anio"] == anio_act) & (df["mes"] == mes_act) & (df["evento"] == "IN")]

reporte = (
    df_mes.groupby(["anio", "mes", "nombre"])
          .size()
          .reset_index(name="asistencias")
          .sort_values(["asistencias", "nombre"], ascending=[False, True])
)

OUT_FILE = OUT_DIR / f"reporte_mensual_{anio_act}_{mes_act:02d}.csv"
reporte.to_csv(OUT_FILE, index=False, encoding="utf-8")

print(f" Reporte mensual generado: {OUT_FILE}")
print(reporte if not reporte.empty else f"Sin asistencias IN en {anio_act}-{mes_act:02d}.")
