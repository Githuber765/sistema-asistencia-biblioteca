import os, pickle, re
from pathlib import Path
import face_recognition as fr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'personas'
OUT  = ROOT / 'data' / 'encodings' / 'personas.pkl'
OUT.parent.mkdir(parents=True, exist_ok=True)

def normalize_name(filename: str) -> str:
    """
    Extrae el nombre completo del archivo en formato:
    <NOMBRES-CON-GUIONES>_<APELLIDOS-CON-GUIONES>_<NUMERO>.jpg
    """
    stem = Path(filename).stem  # sin extensión
    # Quita el número final (por ej. _1 o -2)
    stem = re.sub(r'[_-]\d+$', '', stem)
    # Reemplaza guiones por espacios
    name = stem.replace('-', ' ').replace('_', ' ')
    # Limpia espacios múltiples
    name = re.sub(r'\s+', ' ', name).strip()
    return name

enc_db = {}

for fn in os.listdir(BASE):
    if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = BASE / fn
        name = normalize_name(fn)
        img = fr.load_image_file(str(path))
        locs = fr.face_locations(img, model='hog')
        if not locs:
            print(f'[WARN] No se encontró rostro en: {fn}')
            continue
        enc = fr.face_encodings(img, known_face_locations=locs)[0]
        enc_db.setdefault(name, []).append(enc)
        print(f'[OK] {fn} → {name}')

# Promediar si hay varias fotos de la misma persona
final_db = [{'nombre': n, 'encoding': np.mean(encs, axis=0)} for n, encs in enc_db.items()]

with open(OUT, 'wb') as fh:
    pickle.dump(final_db, fh)

print(f'Guardado: {OUT} ({len(final_db)} personas)')
