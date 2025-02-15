
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# โหลดโมเดลที่เทรนไว้แล้ว
model = joblib.load("Model SVM-upload-No translate.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # โหลดตัวแปลงข้อความที่ใช้ตอนเทรน

# ส่วนของ Streamlit UI
st.title("📊 Sentiment Analysis (SVM) - ภาษาไทย")

# อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("📂 กรุณาอัปโหลดไฟล์ CSV ที่มีคอลัมน์ข้อความ", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ตรวจสอบว่ามีคอลัมน์ "Text" หรือไม่
    if "Text" in df.columns:
        st.write("✅ พบคอลัมน์ข้อความ 'Text'")
        
        # แปลงข้อความเป็นเวกเตอร์
        X = vectorizer.transform(df["Text"])

        # ทำการพยากรณ์ Sentiment
        predictions = model.predict(X)

        # เพิ่มผลลัพธ์เข้า DataFrame
        df["Predicted_Sentiment"] = predictions

        # แสดงตารางข้อมูล
        st.write("### 📌 ผลลัพธ์ที่ได้:")
        st.dataframe(df[["Text", "Predicted_Sentiment"]])

        # ดาวน์โหลดไฟล์ผลลัพธ์
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 ดาวน์โหลดผลลัพธ์เป็น CSV", data=csv, file_name="Sentiment_Predictions.csv", mime="text/csv")

    else:
        st.error("⚠️ ไม่พบคอลัมน์ 'Text' ในไฟล์ กรุณาอัปโหลดไฟล์ที่ถูกต้อง")
