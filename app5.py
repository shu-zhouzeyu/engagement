import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import platform
import os
from matplotlib import font_manager, rcParams

# Plotly Expressをインポート
import plotly.express as px

# 日本語フォント設定 (WordCloudとMatplotlib用)
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/meiryo.ttc'
elif platform.system() == 'Darwin':
    font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
else:
    # Linux (例: Ubuntuの場合) のフォントパス
    font_path = '/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf'
    # 他のLinuxディストリビューションの場合、適宜変更してください

# font_prop はグローバル変数として保持し、Streamlitのキャッシュにも利用
font_prop = None
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
else:
    st.warning(f"指定された日本語フォント '{font_path}' が見つかりません。代替フォントを検索します。")
    font_files = font_manager.findSystemFonts(fontpaths=None)
    found_japanese_font = False
    for f in font_files:
        if "japanese" in f.lower() or "gothic" in f.lower() or "meiryo" in f.lower() or "hiragino" in f.lower() or "noto" in f.lower():
            font_path = f
            font_prop = font_manager.FontProperties(fname=font_path)
            rcParams['font.family'] = font_prop.get_name()
            st.info(f"代替日本語フォント '{font_prop.get_name()}' を設定しました。")
            found_japanese_font = True
            break
    if not found_japanese_font:
        st.error("代替の日本語フォントも見つかりませんでした。文字化けの可能性があります。")
        rcParams['font.family'] = ['sans-serif']

warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from janome.tokenizer import Tokenizer
import re
from itertools import combinations
import json

class SurveyNLPAnalyzer:
    def __init__(self, df):
        self.df = df.copy() # オリジナルDataFrameのコピーを操作
        self.tokenizer = Tokenizer()
        self.analysis_results = {}
        self.question_cols = [col for col in self.df.columns if re.match(r'Q\d+回答', col)]

        self.non_specific_patterns = [
            r'わからない', r'特になし', r'不明', r'なし',
            r'特に.*ない', r'よくわからない', r'^\s*$', r'ない'
        ]
        self.excluded_verbs = {'する', 'なる', 'いる', 'られる', '感じる', 'やすい', 'せる', 'ある', 'いう','自分'}
        
        self.wc_font_path = font_path # Word Cloud用のフォントパス

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text_str = str(text).lower()
        for pattern in self.non_specific_patterns:
            if re.search(pattern, text_str):
                return ""
        text_str = re.sub(r'[！!？?。．、，\s\t　]+', ' ', text_str)
        text_str = re.sub(r'[^\wぁ-んァ-ン一-龥a-zA-Z0-9]', '', text_str)
        return text_str.strip()

    def morphological_analysis(self, text):
        if not text:
            return []
        words = []
        for token in self.tokenizer.tokenize(text):
            base_form = token.base_form
            pos = token.part_of_speech.split(',')[0]
            
            if pos in ['名詞', '動詞', '形容詞']:
                if len(base_form) > 1 and base_form not in self.excluded_verbs:
                    words.append(base_form)
        return words

    def extract_keywords_tfidf(self, texts, top_n=10):
        processed_texts_for_vectorizer = [" ".join(self.morphological_analysis(text)) for text in texts if text]
        
        if not processed_texts_for_vectorizer:
            return []
        
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(processed_texts_for_vectorizer)
        
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]

    def topic_modeling_lda(self, texts, n_topics=5):
        processed_texts_for_vectorizer = [" ".join(self.morphological_analysis(text)) for text in texts if text]

        if len(processed_texts_for_vectorizer) < 2:
            return []
        try:
            vectorizer = TfidfVectorizer(max_features=100, min_df=1, max_df=0.8)
            tfidf_matrix = vectorizer.fit_transform(processed_texts_for_vectorizer)
            
            feature_names = vectorizer.get_feature_names_out()
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf_matrix)
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
                topics.append({'topic_id': topic_idx, 'words': top_words, 'weight': topic.max()})
            return topics
        except Exception as e:
            st.error(f"LDAエラー: {e}")
            return []

    def sentiment_analysis(self, text):
        if not text:
            return {'polarity': 0, 'sentiment': 'neutral'}
        positive_words = ['良い', 'よい', '嬉しい', 'うれしい', '楽しい', '満足', '助かる', '充実']
        negative_words = ['困る', '課題', '不満', '大変', '疲れる', '問題', 'ストレス']
        
        tokens = self.morphological_analysis(text)
        
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        
        if pos_count > neg_count:
            return {'polarity': 1, 'sentiment': 'positive'}
        elif neg_count > pos_count:
            return {'polarity': -1, 'sentiment': 'negative'}
        elif pos_count > 0 and neg_count > 0:
            return {'polarity': 0, 'sentiment': 'ミックス'}
        else:
            return {'polarity': 0, 'sentiment': '中立'}

    def classify_request_type(self, text):
        if not text:
            return 'unknown'
        
        tokens = self.morphological_analysis(text)
        
        maintain_keywords = ['維持', '継続', '現状', '保持', '良い', 'よい', '満足', '嬉しい']
        enhance_keywords = ['強化', '向上', '発展', '拡大', '成長', '進歩', '発達', 'さらに']
        improve_keywords = ['改善', '課題', '問題', '困る', '難しい', '見直し', '変更', '修正']
        
        scores = {
            'maintain': sum(1 for word in tokens if word in maintain_keywords),
            'enhance': sum(1 for word in tokens if word in enhance_keywords),
            'improve': sum(1 for word in tokens if word in improve_keywords)
        }
        
        if all(score == 0 for score in scores.values()):
            return 'maintain'
            
        max_cat = max(scores, key=scores.get)
        return max_cat

    def collocation_analysis(self, texts):
        all_words = []
        for text in texts:
            if text:
                all_words.extend(self.morphological_analysis(text))
        
        if len(all_words) < 2:
            return []
            
        bigrams = [(all_words[i], all_words[i+1]) for i in range(len(all_words) - 1)]
        
        return Counter(bigrams).most_common(10)

    def create_network_graph(self, collocations):
        G = nx.Graph()
        if not collocations:
            return None
        for (w1, w2), count in collocations:
            G.add_edge(w1, w2, weight=count)
        return G if G.number_of_nodes() > 0 else None


    @st.cache_data # 分析結果をキャッシュして高速化
    def analyze_data(_self, df_input, top_n_keywords): # top_n_keywords を引数に追加
        _self.df = df_input.copy() # キャッシュされたdf_inputを使用
        _self.question_cols = [col for col in _self.df.columns if re.match(r'Q\d+回答', col)]

        st.write("### 自然言語処理分析中...")

        all_combined_texts_for_overall_analysis = []
        for col_name in _self.question_cols:
            all_combined_texts_for_overall_analysis.extend(_self.df[col_name].apply(_self.preprocess_text).tolist())
        valid_overall_texts = [text for text in all_combined_texts_for_overall_analysis if text]
        
        st.write(f"全体での有効回答数: {len(valid_overall_texts)}")

        analysis_results = {}
        analysis_results['overall'] = {}
        
        # WordCloudとPlotly Bar Chart用の生の単語リストとDataFrameを保存
        overall_word_list = []
        for text in valid_overall_texts:
            overall_word_list.extend(_self.morphological_analysis(text))
        analysis_results['overall']['word_list'] = overall_word_list
        # ★★★ 修正点: top_n_keywords を Counter.most_common に渡す ★★★
        analysis_results['overall']['top_words'] = Counter(overall_word_list).most_common(top_n_keywords)
        
        # ホバー情報表示のために、キーワードと元のテキストを紐づけたDataFrameを作成
        keyword_data = []
        for col in _self.question_cols:
            for idx, row in _self.df.iterrows():
                preprocessed_text = _self.preprocess_text(row[col])
                if preprocessed_text:
                    words = _self.morphological_analysis(preprocessed_text)
                    for word in words:
                        keyword_data.append({
                            'keyword': word,
                            'original_text': row[col] # ★★★ 修正点: ホバーに表示する内容をoriginal_textのみにする ★★★
                        })
        analysis_results['overall']['detailed_keywords_df'] = pd.DataFrame(keyword_data)

        # TF-IDFキーワード (これは以前の棒グラフ用で、Plotly棒グラフでは'top_words'を使う)
        analysis_results['overall']['tfidf_keywords'] = _self.extract_keywords_tfidf(valid_overall_texts)

        analysis_results['overall']['topics'] = _self.topic_modeling_lda(valid_overall_texts)
        
        overall_sentiments = [_self.sentiment_analysis(t) for t in valid_overall_texts]
        analysis_results['overall']['sentiments'] = Counter([s['sentiment'] for s in overall_sentiments])

        classifications = []
        for i, row in _self.df.iterrows():
            combined_text_parts = []
            for col in _self.question_cols:
                combined_text_parts.append(_self.preprocess_text(row[col]))
            classifications.append(_self.classify_request_type(" ".join(combined_text_parts).strip()))

        temp_df = _self.df.copy() # 一時的に分類結果を追加するためのコピー
        temp_df['classification'] = classifications 
        analysis_results['overall']['classifications'] = Counter(classifications)
        
        _self.df = temp_df # 分類結果を元のDataFrameに反映（Streamlitのセッション管理を考慮）

        analysis_results['overall']['collocations'] = _self.collocation_analysis(valid_overall_texts)

        # 属性別分析
        has_org = '組織' in _self.df.columns and not _self.df['組織'].empty
        has_gender = '性別' in _self.df.columns and not _self.df['性別'].empty

        if has_org:
            org_analysis = _self.df.groupby('組織')['classification'].value_counts().unstack(fill_value=0)
            analysis_results['overall']['org_analysis'] = org_analysis
        else:
            analysis_results['overall']['org_analysis'] = pd.DataFrame()

        if has_gender:
            gender_analysis = _self.df.groupby('性別')['classification'].value_counts().unstack(fill_value=0)
            analysis_results['overall']['gender_analysis'] = gender_analysis
        else:
            analysis_results['overall']['gender_analysis'] = pd.DataFrame()


        analysis_results['questions'] = {}
        for q_col in _self.question_cols:
            st.write(f"--- 設問 '{q_col}' の分析中 ---")
            q_texts = _self.df[q_col].apply(_self.preprocess_text).tolist()
            valid_q_texts = [text for text in q_texts if text]
            st.write(f"設問 '{q_col}' の有効回答数: {len(valid_q_texts)}")

            analysis_results['questions'][q_col] = {}
            
            analysis_results['questions'][q_col]['keywords'] = _self.extract_keywords_tfidf(valid_q_texts)
            
            q_sentiments = [_self.sentiment_analysis(t) for t in valid_q_texts]
            analysis_results['questions'][q_col]['sentiments'] = Counter([s['sentiment'] for s in q_sentiments])
            
            analysis_results['questions'][q_col]['collocations'] = _self.collocation_analysis(valid_q_texts)
            
            q_word_list = [] # ワードクラウドとPlotly Bar Chart用に単語リストを保持
            for text in valid_q_texts:
                q_word_list.extend(_self.morphological_analysis(text))
            # ★★★ 修正点: top_n_keywords を Counter.most_common に渡す ★★★
            analysis_results['questions'][q_col]['top_words'] = Counter(q_word_list).most_common(top_n_keywords)
            analysis_results['questions'][q_col]['word_list'] = q_word_list # ワードクラウド用とPlotlyホバー情報用

            # 設問ごとの詳細キーワードDataFrame (ホバー情報用)
            q_keyword_data = []
            for idx, row in _self.df.iterrows():
                preprocessed_text = _self.preprocess_text(row[q_col])
                if preprocessed_text:
                    words = _self.morphological_analysis(preprocessed_text)
                    for word in words:
                        q_keyword_data.append({
                            'keyword': word,
                            'original_text': row[q_col] # ★★★ 修正点: ホバーに表示する内容をoriginal_textのみにする ★★★
                        })
            analysis_results['questions'][q_col]['detailed_keywords_df'] = pd.DataFrame(q_keyword_data)
        
        return analysis_results, _self.df # 更新されたdfも返す


# Streamlitアプリケーションのメイン関数
def main_app():
    st.set_page_config(layout="wide") # レイアウトをワイドに設定
    st.title("アンケート自由記述分析アプリ")

    # ファイルアップローダー
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    df = None
    if uploaded_file is not None:
        try:
            # CSVを読み込む際にencodingを指定
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.sidebar.success("ファイルが正常にアップロードされました。")
            st.sidebar.dataframe(df.head()) # サイドバーにデータフレームの冒頭を表示

            # ★★★ 新機能: 頻出キーワード数スライダー ★★★
            # defaultは10、最小1、最大50程度、ステップ1
            top_n_keywords = st.sidebar.slider(
                "表示する頻出キーワード数",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )
            st.session_state['top_n_keywords'] = top_n_keywords # セッションに保存

            # Analyzeボタン
            if st.sidebar.button("分析を開始"):
                analyzer = SurveyNLPAnalyzer(df)
                # ★★★ 修正点: analyze_data に top_n_keywords を渡す ★★★
                analysis_results, updated_df = analyzer.analyze_data(df, top_n_keywords) 

                st.session_state['analysis_results'] = analysis_results
                st.session_state['df'] = updated_df # 更新されたDataFrameをセッションに保存
                st.session_state['analyzer'] = analyzer # analyzerインスタンスも保存

        except UnicodeDecodeError:
            st.error("CSVファイルのエンコーディングがUTF-8ではありません。'shift_jis'または'cp932'を試してみてください。")
            try:
                uploaded_file.seek(0) # ファイルポインタを先頭に戻す
                df = pd.read_csv(uploaded_file, encoding='shift_jis')
                st.sidebar.success("ファイルが正常にアップロードされました (Shift-JIS)。")
                st.session_state['df'] = df # Session State に df を保存
                st.info("Shift-JISで読み込みました。分析を開始するには再度「分析を開始」ボタンを押してください。")
            except Exception as e:
                st.error(f"Shift-JISでの読み込みも失敗しました: {e}")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
    else:
        st.info("CSVファイルをアップロードして分析を開始してください。")

    if 'analysis_results' in st.session_state and 'analyzer' in st.session_state:
        analysis_results = st.session_state['analysis_results']
        df = st.session_state['df']
        analyzer = st.session_state['analyzer']
        # ★★★ 修正点: スライダーの値もセッションから取得 ★★★
        top_n_keywords_display = st.session_state.get('top_n_keywords', 10) # デフォルト値は10

        # # デバッグ用: シンプルなグラフでホバー機能の動作確認
        # st.header("--- デバッグ用テストグラフ ---")
        # test_data = pd.DataFrame({
        #     'Category': ['A', 'B', 'C'],
        #     'Value': [10, 20, 50],
        #     'HoverInfo': ['テスト詳細情報1', 'テスト詳細情報2', 'テスト詳細情報3'] # 純粋なテキスト例
        # })

        # test_fig = px.bar(test_data, x='Category', y='Value', 
        #                   title="テストバーチャート",
        #                   custom_data=[test_data['HoverInfo'].tolist()]) 

        # test_fig.update_traces(
        #     hovertemplate="<b>カテゴリ:</b> %{x}<br>" +
        #                   "<b>値:</b> %{y}<br>" +
        #                   "<b>詳細:</b> %{customdata}" + 
        #                   "<extra></extra>" # これでツールチップに余計なTrace情報が表示されなくなる
        # )
        # st.plotly_chart(test_fig, use_container_width=True)
        # st.header("----------------------------")

        st.header("✨ 全体分析結果")

        # 全体結果の表示 (2列レイアウト)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("要望分類分布")
            classification_counts = analysis_results['overall'].get('classifications', Counter())
            if classification_counts and any(v > 0 for v in classification_counts.values()):
                labels = list(classification_counts.keys())
                values = list(classification_counts.values())
                fig_pie = px.pie(names=labels, values=values, title='要望分類分布 (全体)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("要望分類データなし (全体)")

            st.subheader("感情分析結果")
            sentiment_counts = analysis_results['overall'].get('sentiments', Counter())
            if sentiment_counts and any(sentiment_counts.values()):
                sent_df = pd.DataFrame(sentiment_counts.items(), columns=['感情', '件数'])
                fig_sentiment = px.bar(sent_df, x='感情', y='件数', title='感情分析結果 (全体)')
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("感情分析データなし (全体)")

            st.subheader("組織別要望分類")
            org_analysis = analysis_results['overall'].get('org_analysis', pd.DataFrame())
            if not org_analysis.empty and not org_analysis.sum().sum() == 0:
                # Plotlyで積み上げ棒グラフ
                org_analysis_melted = org_analysis.reset_index().melt(id_vars='組織', var_name='分類', value_name='件数')
                fig_org = px.bar(org_analysis_melted, x='組織', y='件数', color='分類', title='組織別要望分類 (全体)')
                st.plotly_chart(fig_org, use_container_width=True)
            else:
                st.info("組織別分類データなし (全体)")

        with col2:
            st.subheader(f"頻出キーワード上位{top_n_keywords_display}件") # スライダーの値をタイトルに反映
            overall_top_words_df = pd.DataFrame(analysis_results['overall'].get('top_words', []), columns=['キーワード', '出現回数'])
            if not overall_top_words_df.empty:
                detailed_df = analysis_results['overall']['detailed_keywords_df']
                
                # ★★★ 修正点: ホバーテキストを純粋なoriginal_textのみにする ★★★
                # 各キーワードに関連するoriginal_textをすべて結合する
                overall_top_words_df['custom_hover_text'] = overall_top_words_df['キーワード'].apply(
                    lambda k: "-".join([
                        row['original_text'] # <-- これが純粋なCSV抽出文言
                        for _, row in detailed_df[detailed_df['keyword'] == k].iterrows()
                    ]) or "該当する回答がありません。"
                )
                
                # --- デバッグ用出力 ---
                # st.write("--- Overall Custom Hover Text (Sample) ---")
                # st.dataframe(overall_top_words_df[['キーワード', 'custom_hover_text']].head())
                # --- デバッグ用出力終わり ---

                fig_keywords = px.bar(overall_top_words_df, x='出現回数', y='キーワード', orientation='h', 
                                    title=f'頻出キーワード上位{top_n_keywords_display} (全体)', # スライダーの値をタイトルに反映
                                    custom_data=[overall_top_words_df['custom_hover_text'].tolist()]) 
                
                fig_keywords.update_traces(
                    hovertemplate="<b>%{y}</b><br>出現回数: %{x}<br>" +
                                  "<b>関連する回答:</b><br>%{customdata}" + 
                                  "<extra></extra>" 
                )
                
                st.plotly_chart(fig_keywords, use_container_width=True)
            else:
                st.info(f"頻出キーワードなし (全体) - 現在のキーワード数設定: {top_n_keywords_display}")

            st.subheader("性別別要望分類")
            gender_analysis = analysis_results['overall'].get('gender_analysis', pd.DataFrame())
            if not gender_analysis.empty and not gender_analysis.sum().sum() == 0:
                gender_analysis_melted = gender_analysis.reset_index().melt(id_vars='性別', var_name='分類', value_name='件数')
                fig_gender = px.bar(gender_analysis_melted, x='性別', y='件数', color='分類', title='性別別要望分類 (全体)')
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("性別分類データなし (全体)")
            

        st.markdown("---")

        st.header("📝 設問別分析結果")

        # 設問ごとの結果を表示
        if analyzer.question_cols: 
            for q_col in analyzer.question_cols:
                st.subheader(f"### {q_col} の分析結果")
                q_results = analysis_results['questions'][q_col]

                col_q1, col_q2, col_q3 = st.columns(3)

                with col_q1:
                    st.write(f"#### 頻出キーワード上位{top_n_keywords_display}件") # スライダーの値をタイトルに反映
                    q_top_words_df = pd.DataFrame(q_results.get('top_words', []), columns=['キーワード', '出現回数'])
                    if not q_top_words_df.empty:
                        detailed_df = q_results['detailed_keywords_df']
                        
                        # ★★★ 修正点: ホバーテキストを純粋なoriginal_textのみにする ★★★
                        q_top_words_df['custom_hover_text'] = q_top_words_df['キーワード'].apply(
                            lambda k: "-".join([
                                row['original_text'] # <-- これが純粋なCSV抽出文言
                                for _, row in detailed_df[detailed_df['keyword'] == k].iterrows()
                            ]) or "該当する回答がありません。"
                        )

                        # --- デバッグ用出力 ---
                        st.write(f"--- {q_col} Custom Hover Text (Sample) ---") 
                        st.dataframe(q_top_words_df[['キーワード', 'custom_hover_text']].head())
                        # --- デバッグ用出力終わり ---

                        fig_q_keywords = px.bar(q_top_words_df, x='出現回数', y='キーワード', orientation='h',
                                                title=f'頻出キーワード上位{top_n_keywords_display} ({q_col})', # スライダーの値をタイトルに反映
                                                custom_data=[q_top_words_df['custom_hover_text'].tolist()]) 

                        fig_q_keywords.update_traces(
                            hovertemplate="<b>%{y}</b><br>出現回数: %{x}<br>" +
                                          "<b>関連する回答:</b><br>%{customdata}" + 
                                          "<extra></extra>"
                        )
                        st.plotly_chart(fig_q_keywords, use_container_width=True)
                    else:
                        st.info(f"頻出キーワードなし ({q_col}) - 現在のキーワード数設定: {top_n_keywords_display}")

                with col_q2:
                    st.write("#### 感情分析結果")
                    q_sentiment_counts = q_results.get('sentiments', Counter())
                    if q_sentiment_counts and any(q_sentiment_counts.values()):
                        q_sent_df = pd.DataFrame(q_sentiment_counts.items(), columns=['感情', '件数'])
                        fig_q_sentiment = px.bar(q_sent_df, x='感情', y='件数', title=f'感情分析結果 ({q_col})')
                        st.plotly_chart(fig_q_sentiment, use_container_width=True)
                    else:
                        st.info(f"感情分析データなし ({q_col})")

                st.markdown("---") # 各設問の間に区切り線
        else: # question_cols が空の場合のメッセージ
            st.info("分析対象の設問回答列（Q〇〇回答）が見つかりませんでした。CSVファイルの列名をご確認ください。")


if __name__ == '__main__':
    main_app()