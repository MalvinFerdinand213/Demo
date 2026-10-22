import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from model import predict_aspect_scores, get_star_tier_benchmark, ASPECTS

st.set_page_config(page_title="Multimodal Hotel Review Analysis", layout="wide")

st.title("🏨 Multimodal Aspect-Based Hotel Review Analysis")
st.markdown("Mini predictive deployment & comparison analysis berdasarkan tingkatan bintang hotel.")

# --- SIDEBAR / INPUT SECTION ---
st.sidebar.header("📥 Input Review Data")
review_text = st.sidebar.text_area("Hotel Review Text", "Kamarnya cukup bersih dan pelayanannya ramah, tapi makanan sarapannya kurang bervariasi.")
uploaded_file = st.sidebar.file_uploader("Upload Hotel Image", type=["jpg", "jpeg", "png"])
star_rating = st.sidebar.slider("Hotel Star Tier", min_value=1, max_value=5, value=4)

submit_btn = st.sidebar.button("Analyze Review")

# --- MAIN CONTENT ---
if submit_btn:
    if not review_text:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = Image.new('RGB', (200, 200), color=(200, 200, 200))

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Submitted Content")
            st.write(f"**Star Rating:** {'⭐' * star_rating} ({star_rating}-Star Hotel)")
            st.write(f"**Review:** {review_text}")
            if uploaded_file:
                st.image(image, caption="Uploaded Image", use_container_width=True)

        predictions = predict_aspect_scores(review_text, image, star_rating)
        benchmarks = get_star_tier_benchmark(star_rating)

        with col2:
            st.subheader("🎯 Step 1: Per-Aspect Sentiment & Score")
            
            df_res = pd.DataFrame([
                {
                    "Aspect": aspect,
                    "Score (1-5)": data["score"],
                    "Polarity": data["polarity"]
                }
                for aspect, data in predictions.items()
            ])
            st.dataframe(df_res, use_container_width=True)

        st.divider()

        st.subheader("📊 Step 2: Hotel Star-Tier Differentiation Analysis")
        st.caption(f"Membandingkan hasil ulasan dengan standar rata-rata Hotel Bintang {star_rating}.")

        categories = ASPECTS
        user_scores = [predictions[a]["score"] for a in categories]
        benchmark_scores = [benchmarks[a] for a in categories]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=user_scores,
            theta=categories,
            fill='toself',
            name='Current Review Score'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=benchmark_scores,
            theta=categories,
            fill='toself',
            name=f'Avg {star_rating}-Star Hotel Benchmark'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            title=f"Review Score vs {star_rating}-Star Hotel Average"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Key Differentiation Insights")
        insights = []
        for a in categories:
            diff = predictions[a]["score"] - benchmarks[a]
            if diff >= 0.5:
                insights.append(f"- **{a}**: Performa **unggul** (+{round(diff,2)}) dibanding rata-rata hotel bintang {star_rating}.")
            elif diff <= -0.5:
                insights.append(f"- **{a}**: Performa **di bawah standar** ({round(diff,2)}) untuk kelas hotel bintang {star_rating}.")

        if insights:
            for insight in insights:
                st.write(insight)
        else:
            st.write(f"Performa ulasan ini relatif **seimbang** dengan standar hotel bintang {star_rating}.")
else:
    st.info("Silakan isi form di sebelah kiri lalu klik **Analyze Review** untuk melihat hasil analisis.")
