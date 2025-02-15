
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# โหลดโมเดลที่เทรนไว้แล้ว
model = joblib.load("Model SVM-upload-No translate.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # โหลดตัวแปลงข้อความที่ใช้ตอนเทรน

# ส่วนของ Streamlit UI
st.title("📊 Sentiment Analysis (SVM) - วิเคราะห์ความคิดเห็นลูกค้า")

# เลือกโหมดการใช้งาน: ป้อนข้อความเอง หรือ อัปโหลดไฟล์
option = st.radio("เลือกวิธีการวิเคราะห์:", ["📝 ป้อนข้อความเอง", "📂 อัปโหลดไฟล์ Excel"])

# 🔹 กรณีป้อนข้อความเอง
if option == "📝 ป้อนข้อความเอง":
    user_input = st.text_area("✏️ กรุณาป้อนข้อความที่ต้องการวิเคราะห์:")
    
    if st.button("🔍 วิเคราะห์ Sentiment"):
        if user_input.strip():  # ตรวจสอบว่ามีข้อความป้อนเข้ามาหรือไม่
            X_input = vectorizer.transform([user_input])  # แปลงข้อความเป็นเวกเตอร์
            prediction = model.predict(X_input)[0]  # ทำนายผล
            
            # แสดงผลการวิเคราะห์
            st.subheader("📌 ผลการวิเคราะห์ Sentiment")
            st.write(f"**ความคิดเห็นที่ป้อน:** {user_input}")
            st.write(f"🎯 **ผลลัพธ์:** `{prediction}`")
        else:
            st.error("⚠️ กรุณาป้อนข้อความก่อนทำการวิเคราะห์")

# 🔹 กรณีอัปโหลดไฟล์ Excel
elif option == "📂 อัปโหลดไฟล์ Excel":
    uploaded_file = st.file_uploader("📂 กรุณาอัปโหลดไฟล์ Excel (`.xlsx`)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)  # อ่านไฟล์ Excel
        
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

