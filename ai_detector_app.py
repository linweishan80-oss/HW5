# ai_detector_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# 1. 建立示範資料集（之後可換成自己的資料）
# =========================
def build_demo_dataset():
    """
    label = 0 -> Human
    label = 1 -> AI
    先用很小的示範資料讓整個流程可以跑通。
    之後如果有自己的資料，可以直接改這個 function。
    """
    human_texts = [
        "今天上課的時候老師講了很多例子，其實我有一點聽不太懂，但回家再看一次應該就可以了。",
        "昨天跟朋友去夜市吃東西，人超級多，結果排隊排到腳很酸。",
        "我覺得寫作最難的地方是要把自己的想法整理清楚，還要讓別人看得懂。",
        "前幾天突然下大雨，結果我忘記帶傘，全身都淋濕，只好趕快回家洗澡換衣服。",
        "這學期的作業有點多，有時候會覺得壓力很大，但慢慢做其實還是可以完成。"
    ]

    ai_like_texts = [
        "This paragraph is generated to demonstrate the style of AI-written content, which often appears fluent and well structured.",
        "In recent years, artificial intelligence has significantly improved, enabling models to produce coherent and context-aware text.",
        "The purpose of this text is to resemble machine-generated language, with formal tone and generic statements.",
        "AI-generated content typically maintains consistent grammar and uses relatively neutral expressions throughout the paragraph.",
        "Modern language models are capable of generating long passages that sound natural, even without deep understanding of the topic."
    ]

    texts = human_texts + ai_like_texts
    labels = [0] * len(human_texts) + [1] * len(ai_like_texts)  # 0=Human, 1=AI

    df = pd.DataFrame({"text": texts, "label": labels})
    return df


# =========================
# 2. 訓練模型（用 cache 避免每次重訓）
# =========================
@st.cache_resource
def train_model():
    """
    訓練 TF-IDF + Logistic Regression。
    這裡用示範資料。之後若有自己的 dataset，
    可以在這裡改成讀 CSV 再訓練。
    """
    df = build_demo_dataset()
    X = df["text"].values
    y = df["label"].values  # 0 = Human, 1 = AI

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    # 簡單做一下 demo accuracy，回傳給前端顯示
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return pipeline, acc


# =========================
# 3. Streamlit 介面
# =========================
st.set_page_config(
    page_title="AI / Human 文章偵測器",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI / Human 文章偵測器")
st.write(
    """
    請輸入一段文章，我會根據訓練好的模型，估計這段文字
    比較像是 **AI 產生** 還是 **人類撰寫**。
    
    > ⚠️ 這只是示範級小模型，使用很少量資料訓練，  
    > 只能當作作業 / 練習用，不代表真實 AI 偵測器的可靠度。
    """
)

with st.expander("模型資訊（demo 用）", expanded=False):
    st.write("本頁面使用：TF-IDF + Logistic Regression")
    st.write("訓練資料：簡單手刻 5 筆 human + 5 筆 AI 風格句子")

# 先訓練 / 載入模型（只會在第一次呼叫時跑）
with st.spinner("初始化模型中..."):
    model, demo_acc = train_model()

st.caption(f"（Demo 小測試集準確率約為：{demo_acc*100:.1f}%）")

text = st.text_area("✏️ 請貼上要檢測的文本：", height=220)

col_run1, col_run2 = st.columns([1, 1])
with col_run1:
    auto_run = st.checkbox("輸入文字就自動分析", value=True)
with col_run2:
    run_button = st.button("開始偵測")

should_run = False
if auto_run:
    should_run = bool(text.strip())
else:
    should_run = run_button and bool(text.strip())

if should_run:
    with st.spinner("分析中..."):
        proba = model.predict_proba([text])[0]  # shape: (2,)
        classes = list(model.classes_)          # e.g. [0, 1] where 1 = AI

        # 假設：0 = Human, 1 = AI
        ai_index = classes.index(1)
        human_index = classes.index(0)

        ai_prob = float(proba[ai_index])
        human_prob = float(proba[human_index])

    st.subheader("📊 判斷結果")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("AI 產生機率", f"{ai_prob * 100:.1f}%")
    with col2:
        st.metric("Human 撰寫機率", f"{human_prob * 100:.1f}%")

    st.write("---")
    st.write("機率視覺化：")

    # 簡單條狀圖
    st.bar_chart({
        "AI": [ai_prob],
        "Human": [human_prob]
    })

elif text.strip() == "" and run_button:
    st.warning("請先輸入一些文本再按「開始偵測」。")

