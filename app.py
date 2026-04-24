import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib, uuid
import pandas as pd
from supabase import create_client, Client

BUCKET_NAME = "uploads"
TABLE_NAME  = "predictions"


@st.cache_resource
def get_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((8, 8), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    arr = arr / arr.max() * 16.0
    return arr.flatten().reshape(1, -1)


def save_blob(sb: Client, image_bytes: bytes, filename: str) -> str:
    sb.storage.from_(BUCKET_NAME).upload(
        path=filename,
        file=image_bytes,
        file_options={"content-type": "image/png", "upsert": "true"},
    )
    return sb.storage.from_(BUCKET_NAME).get_public_url(filename)


def save_table(sb: Client, filename: str, digit: int,
               confidence: float, blob_url: str):
    sb.table(TABLE_NAME).insert({
        "filename":   filename,
        "digit":      digit,
        "confidence": round(confidence, 4),
        "blob_url":   blob_url,
    }).execute()


def load_history(sb: Client) -> pd.DataFrame:
    response = (
        sb.table(TABLE_NAME)
        .select("created_at, digit, confidence, filename, blob_url")
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["confidence"] = (df["confidence"] * 100).round(2).astype(str) + "%"
    df["imagen"] = df["blob_url"].apply(
        lambda url: f'<a href="{url}" target="_blank">🔗 ver</a>'
    )
    df = df.drop(columns=["blob_url", "filename"])
    df = df.rename(columns={
        "created_at": "Fecha",
        "digit":      "Dígito",
        "confidence": "Confianza",
        "imagen":     "Imagen",
    })
    return df


# ── UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="centered")
st.title("🔢 Clasificador de Dígitos - UP")
st.caption("Sube una imagen de un número escrito a mano (0–9)")

sb  = get_supabase()
clf = load_model()

uploaded = st.file_uploader("Imagen PNG o JPG", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Imagen subida", use_container_width=True)

    arr        = preprocess(img)
    pred       = int(clf.predict(arr)[0])
    proba      = clf.predict_proba(arr)[0]
    confidence = float(proba[pred])

    with col2:
        st.metric("Dígito predicho", str(pred))
        st.metric("Confianza", f"{confidence:.1%}")
        st.bar_chart({str(i): proba[i] for i in range(10)})

    filename = f"{uuid.uuid4()}_{uploaded.name}"

    with st.spinner("Guardando en Supabase..."):
        blob_url = save_blob(sb, uploaded.getvalue(), filename)
        save_table(sb, filename, pred, confidence, blob_url)

    st.success("✅ Imagen guardada en Storage · Predicción registrada en tabla")

    with st.expander("Detalles"):
        st.json({
            "digit":      pred,
            "confidence": confidence,
            "blob_url":   blob_url,
            "filename":   filename,
        })

    # ── Historial ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Historial de predicciones")

    try:
        df = load_history(sb)
        if df.empty:
            st.info("Aún no hay predicciones registradas.")
        else:
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.caption(f"{len(df)} predicciones · últimas 50")
    except Exception as e:
        st.error(f"Error cargando historial: {e}")
