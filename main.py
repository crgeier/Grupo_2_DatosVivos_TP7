from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder

# ==============================
# Cargar modelo y columnas (RUTAS CORREGIDAS)
# ==============================

MODEL_PATH = "C:/Users/Jorge Santacecilia/Documents/Workplace/TP7/rf.pkl"
COLUMNS_PATH = "C:/Users/Jorge Santacecilia/Documents/Workplace/TP7/categories_ohe.pickle"

# Cargar modelo entrenado
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Cargar nombres de columnas usadas para entrenar
with open(COLUMNS_PATH, "rb") as handle:
    feature_columns = pickle.load(handle)

# ==============================
# Cargar bins de discretización (RUTAS CORREGIDAS)
# ==============================
with open("C:/Users/Jorge Santacecilia/Documents/Workplace/TP7/saved_bins_ph.pickle", "rb") as f:
    saved_bins_ph = pickle.load(f)
with open("C:/Users/Jorge Santacecilia/Documents/Workplace/TP7/saved_bins_sulfate.pickle", "rb") as f:
    saved_bins_sulfate = pickle.load(f)
with open("C:/Users/Jorge Santacecilia/Documents/Workplace/TP7/saved_bins_trihalomethanes.pickle", "rb") as f:
    saved_bins_trihalomethanes = pickle.load(f)


# ==============================
# Inicializar la API
# ==============================

app = FastAPI(title="API Predicción Potabilidad del Agua")

# ==============================
# Definir estructura de entrada
# ==============================
class WaterData(BaseModel):
    ph: float | None = None
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float | None = None
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float | None = None
    Turbidity: float
# ==============================
# Endpoints
# ==============================
@app.get("/")
async def root():
    return {"message": "API para predecir potabilidad del agua"}

@app.post("/prediccion")
def predict_potability(data: WaterData):
    global ultima_respuesta, ultimos_datos
    # Convertir a dict → DataFrame
    data_dict = jsonable_encoder(data)
    df_input = pd.DataFrame([data_dict])

    # ==============================
    # 1 Discretizar variables
    # ==============================
    # ph
    df_input["ph_disc"] = pd.cut(df_input["ph"], bins=saved_bins_ph, include_lowest=True, duplicates="drop")
    # sulfate
    df_input["Sulfate_disc"] = pd.cut(df_input["Sulfate"], bins=saved_bins_sulfate, include_lowest=True, duplicates="drop")
    # trihalomethanes
    df_input["Trihalomethanes_disc"] = pd.cut(df_input["Trihalomethanes"], bins=saved_bins_trihalomethanes, include_lowest=True, duplicates="drop")

    # ==============================
    # 2 Reemplazar NaN con 'desconocido'
    # ==============================
    df_input["ph_disc"] = df_input["ph_disc"].cat.add_categories("desconocido").fillna("desconocido")
    df_input["Sulfate_disc"] = df_input["Sulfate_disc"].cat.add_categories("desconocido").fillna("desconocido")
    df_input["Trihalomethanes_disc"] = df_input["Trihalomethanes_disc"].cat.add_categories("desconocido").fillna("desconocido")

    # ==============================
    # 3 One Hot Encoding
    # ==============================
    dummies_ph = pd.get_dummies(df_input["ph_disc"], prefix="ph")
    dummies_sulfate = pd.get_dummies(df_input["Sulfate_disc"], prefix="sulfate")
    dummies_trihalo = pd.get_dummies(df_input["Trihalomethanes_disc"], prefix="trihalomethanes")

    df_final = pd.concat([df_input, dummies_ph, dummies_sulfate, dummies_trihalo], axis=1)

    # Eliminar las columnas discretizadas originales
    df_final = df_final.drop(["ph_disc", "Sulfate_disc", "Trihalomethanes_disc"], axis=1)

    # ==============================
    # 4 Alinear columnas con el modelo
    # ==============================
    df_ready = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in df_final.columns:
            df_ready[col] = df_final[col]
        else:
            df_ready[col] = 0

    # ==============================
    # 5 Predicción
    # ==============================
    prediction = model.predict(df_ready)[0]
    probability = model.predict_proba(df_ready)[0][1]

    # Guardar la última respuesta
    ultima_respuesta = {
        "prediccion": int(prediction),
        "probabilidad_potable": round(float(probability), 4)
    }
    ultimos_datos = data_dict

    return ultima_respuesta
@app.get("/prediccion")
async def predict_get():
    return {
        "ultima_respuesta_del_modelo": ultima_respuesta,
        "ultimos_datos_enviados": ultimos_datos
    }

# ===============================
# Ejecutar API (solo si se corre local)
# ===============================
# Corre en http://127.0.0.1:8000 o http://0.0.0.0:8000
if __name__ == '__main__':
    # 0.0.0.0 o 127.0.0.1
    uvicorn.run(app, host='127.0.0.1', port=8000)