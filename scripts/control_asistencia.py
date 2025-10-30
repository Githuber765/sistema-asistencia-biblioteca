import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2, pickle, csv
import numpy as np
import face_recognition as fr
from datetime import datetime, date
from utils.io import ensure_csv, append_csv

# ===================== Configuración =====================
ENC_FILE   = os.path.join('data','encodings','personas.pkl')
CSV_FILE   = os.path.join('data','asistencia.csv')
TOL        = 0.6
CAM_ID     = 'Entrada_Biblioteca'
WIN_TITLE  = 'Asistencia - Biblioteca'

ensure_csv(CSV_FILE, ['timestamp','nombre','cam','evento','dist'])

# ===================== Base facial =====================
with open(ENC_FILE, 'rb') as fh:
    db = pickle.load(fh)
known_enc = [d['encoding'] for d in db]
names     = [d['nombre']   for d in db]

# ===================== Estado del día =====================
def cargar_ultimo_evento_hoy(csv_path):
    hoy = datetime.now().date()
    ult = {}  # nombre -> 'IN' | 'OUT'
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fecha = row['timestamp'].split(' ')[0]
                if fecha == str(hoy):
                    ult[row['nombre']] = row['evento']
    except FileNotFoundError:
        pass
    return ult

ultimo_evento_hoy = cargar_ultimo_evento_hoy(CSV_FILE)
dia_cache = date.today()

def puede_registrar(nombre, modo, ultimo_evento):
    """
    Reglas:
    - IN: bloquea si el último de hoy ya fue IN.
    - OUT: bloquea si el último de hoy ya fue OUT o si no hubo IN previo hoy.
    - NONE: nunca registra.
    """
    if modo == 'NONE':
        return False, 'modo NONE'
    if modo == 'IN':
        if ultimo_evento == 'IN':
            return False, 'IN duplicado hoy'
        return True, None
    if modo == 'OUT':
        if ultimo_evento == 'OUT':
            return False, 'OUT duplicado hoy'
        if ultimo_evento is None:
            return False, 'OUT sin IN previo hoy'
        return True, None
    return False, 'modo desconocido'

# ===================== Modo por teclado =====================
modo = 'NONE'  # arranca neutral: 'IN' | 'OUT' | 'NONE'

def set_modo_from_key(k):
    global modo
    if k == ord('i'):
        modo = 'IN'
    elif k == ord('o'):
        modo = 'OUT'
    elif k == ord('n'):
        modo = 'NONE'

# ===================== Cámara =====================
cap = cv2.VideoCapture(0)
print("Teclas: I=IN, O=OUT, N=None, Q=salir")
cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Si cambia el día mientras el programa está abierto, recargar cache
    if date.today() != dia_cache:
        ultimo_evento_hoy = cargar_ultimo_evento_hoy(CSV_FILE)
        dia_cache = date.today()

    # Leer teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    set_modo_from_key(key)

    # Reconocimiento facial
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = fr.face_locations(rgb, model='hog')
    encs = fr.face_encodings(rgb, locs)

    # Para mostrar "Estado actual" arriba a la izquierda
    reconocidos_en_frame = []  # lista de nombres reconocidos (únicos) en este frame

    for (top, right, bottom, left), enc in zip(locs, encs):
        label = 'Desconocido'
        color = (0, 255, 255)
        min_d = 1.0

        if known_enc:
            dists = fr.face_distance(known_enc, enc)
            idx   = int(np.argmin(dists))
            min_d = float(dists[idx])

            if min_d <= TOL:
                name  = names[idx]
                label = f'{name} ({min_d:.2f})'
                color = (0, 200, 0)

                # Intento de registro según modo y validaciones
                ultimo = ultimo_evento_hoy.get(name)  # None | 'IN' | 'OUT'
                ok_reg, motivo = puede_registrar(name, modo, ultimo)

                if ok_reg:
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    append_csv(CSV_FILE, [ts, name, CAM_ID, modo, f'{min_d:.4f}'])
                    ultimo_evento_hoy[name] = modo
                    print(f'[LOG] {ts} {name} {modo} {min_d:.4f}')
                else:
                    # No registró (modo NONE, duplicado o OUT sin IN)
                    # Log informativo, sin overlay adicional
                    print(f'[SKIP] {name} {modo} -> {motivo}')

                reconocidos_en_frame.append(name)

        # Dibujo por rostro
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (left + 4, bottom - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ---------- Overlays globales (arriba a la izquierda) ----------
    # Línea 1: MODO (amarillo)
    cv2.putText(frame, f"Modo: {modo}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Línea 2: ESTADO ACTUAL (según lo reconocido en este frame)
    estado_text = "Estado actual: ---"
    estado_color = (160, 160, 160)  # gris por defecto

    if len(set(reconocidos_en_frame)) == 1:
        persona = list(set(reconocidos_en_frame))[0]
        estado = ultimo_evento_hoy.get(persona)
        if estado == 'IN':
            estado_text = "Estado actual: IN"
            estado_color = (0, 255, 0)  # verde
        elif estado == 'OUT':
            estado_text = "Estado actual: OUT"
            estado_color = (0, 0, 255)  # rojo
        else:
            estado_text = "Estado actual: ---"
            estado_color = (160, 160, 160)
    elif len(set(reconocidos_en_frame)) > 1:
        estado_text = "Estado actual: Varios"
        estado_color = (160, 160, 160)

    cv2.putText(frame, estado_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, estado_color, 2)

    # Ayuda de atajos cuando está en NONE
    if modo == 'NONE':
        cv2.putText(frame, "I=IN  O=OUT  N=None  Q=Salir", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow(WIN_TITLE, frame)

cap.release()
cv2.destroyAllWindows()
