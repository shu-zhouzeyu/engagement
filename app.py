# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import re

st.title("📝 自由記述アンケート分析アプリ")

# --- 日本語フォント（WordCloud用）
FONT_PATH = "./static/fonts/ヒラギノ角ゴ ProN W6.ttc"  # 適宜調整

# --- トークナイザと分類語彙
tokenizer = Tokenizer()
excluded_verbs = {'する', 'なる', 'いる', 'られる', '感じる', 'やすい', 'せる', 'ある'}
non_specific = re.compile(r"^\s*$|わからない|特になし|なし|ない|不明")
category_keywords = {
    '維持': ['現状維持', 'このまま', '満足', '継続', '良い', 'ありがたい'],
    '強化': ['もっと', 'さらに', '強化', '拡大', '増やす', '深める'],
    '改善': ['改善', '変える', '課題', '直す', '対応', '不満', 'やめて', '困る']
}
positive_words = {'嬉しい', '良い', 'やりがい', '楽しい', '満足', '助かる', '充実'}
negative_words = {'困る', '課題', '不満', '大変', '疲れる', '問題', 'ストレス'}

# --- 関数
def tokenize(text):
    if pd.isna(text) or non_specific.match(text):
        return []
    text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', str(text))
    tokens = []
    for token in tokenizer.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '動詞', '形容詞'] and base not in excluded_verbs and len(base) > 1:
            tokens.append(base)
    return tokens

def classify_category(tokens):
    matched = {cat for cat, words in category_keywords.items() if any(t in words for t in tokens)}
    if not matched:
        return '不明'
    elif len(matched) == 1:
        return list(matched)[0]
    else:
        return 'ミックス'

def judge_sentiment(tokens):
    pos = any(token in positive_words for token in tokens)
    neg = any(token in negative_words for token in tokens)
    if pos and not neg:
        return 'ポジティブ'
    elif neg and not pos:
        return 'ネガティブ'
    elif pos and neg:
        return 'ミックス'
    else:
        return '中立'

def show_wordcloud(counter, title):
    wc = WordCloud(font_path=FONT_PATH, background_color='white', width=800, height=400)
    img = wc.generate_from_frequencies(counter)
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- ファイル読み込み
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Q1_tokens'] = df['Q1回答'].apply(tokenize)
    df['Q2_tokens'] = df['Q2回答'].apply(tokenize)
    df['Q3_tokens'] = df['Q3回答'].apply(tokenize)
    df['all_tokens'] = df['Q1_tokens'] + df['Q2_tokens'] + df['Q3_tokens']

    df['category'] = df['all_tokens'].apply(classify_category)
    df['sentiment'] = df['all_tokens'].apply(judge_sentiment)

    # --- 属性選択
    org_col = df.columns[3]
    age_col = df.columns[4]
    gender_col = df.columns[5]

    st.sidebar.header("📂 フィルタリング")
    org = st.sidebar.selectbox("組織", ['すべて'] + df[org_col].dropna().unique().tolist())
    age = st.sidebar.selectbox("年代", ['すべて'] + df[age_col].dropna().unique().tolist())
    gender = st.sidebar.selectbox("性別", ['すべて'] + df[gender_col].dropna().unique().tolist())

    filtered_df = df.copy()
    if org != 'すべて':
        filtered_df = filtered_df[filtered_df[org_col] == org]
    if age != 'すべて':
        filtered_df = filtered_df[filtered_df[age_col] == age]
    if gender != 'すべて':
        filtered_df = filtered_df[filtered_df[gender_col] == gender]

    st.write(f"⚙️ フィルタ後のデータ件数: {len(filtered_df)} 件")

    # --- 可視化: 感情分布
    st.subheader("📊 感情傾向（円グラフ）")
    sentiment_counts = filtered_df['sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    ax1.axis('equal')
    st.pyplot(fig1)

    # --- カテゴリ別頻出単語
    st.subheader("🔍 分類別頻出単語")
    for cat in ['維持', '強化', '改善']:
        cat_df = filtered_df[filtered_df['category'] == cat]
        tokens = [token for tokens in cat_df['all_tokens'] for token in tokens]
        counter = Counter(tokens)
        if counter:
            show_wordcloud(counter, f"{cat}カテゴリのWordCloud（Top Words）")

    # --- 元の記述例
    st.subheader("🧾 要望例（元文付き）")
    ex_cat = st.selectbox("表示するカテゴリ", ['維持', '強化', '改善', 'ミックス', '不明'])
    ex_df = filtered_df[filtered_df['category'] == ex_cat].head(10)
    for idx, row in ex_df.iterrows():
        st.markdown(f"- **ID {idx}**: Q1: {row['Q1回答']} / Q2: {row['Q2回答']} / Q3: {row['Q3回答']}")

    # --- 結果の保存
    st.subheader("💾 分析結果の保存")
    result_df = filtered_df.copy()
    csv = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 CSVをダウンロード", csv, file_name='analyzed_output.csv', mime='text/csv')