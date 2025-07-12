import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import platform
import os
from matplotlib import font_manager, rcParams

# WordCloudライブラリをインポート
from wordcloud import WordCloud

# Plotly Expressをインポート
import plotly.express as px

# NetworkXをインポート (グラフ描画のため)
import networkx as nx

# Streamlit Plotly Eventsをインポート
from streamlit_plotly_events import plotly_events

# 日本語フォント設定 (WordCloudとMatplotlib用)

# アプリケーションディレクトリ内のフォントパスを指定
# 例えば、カレントディレクトリに 'fonts' フォルダがあり、その中に 'NotoSansJP-Regular.ttf' がある場合
font_file_name = 'NotoSansJP-VariableFont_wght.ttf' # または使用したいNoto Sans JPのファイル名

# fonts フォルダ内にあるか、または直接カレントディレクトリにあるかを確認
if os.path.exists(f'fonts/{font_file_name}'):
    font_path = f'fonts/{font_file_name}'
elif os.path.exists(font_file_name): # 直接カレントディレクトリにある場合
    font_path = font_file_name
else:
    # 最終的なフォールバック（デプロイ環境では通常このパスは存在しないはずですが、念のため）
    # ローカル開発時の参考用として残しておく
    st.warning(f"指定された同梱フォント '{font_file_name}' が見つかりません。システムフォントを試します。")
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/meiryo.ttc'
    elif platform.system() == 'Darwin':
        font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
    else:
        font_path = '/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf'


font_prop = None
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    st.success(f"フォント '{font_prop.get_name()}' を設定しました。")
else:
    st.error(f"指定された日本語フォント '{font_path}' が見つかりません。文字化けの可能性があります。")
    # ここに到達した場合、フォントファイルが見つからなかったことを明確にする
    rcParams['font.family'] = ['sans-serif'] # 最終的なフォールバック
        
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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
        self.excluded_verbs = {'する', 'なる', 'いる', 'られる', '感じる', 'やすい', 'せる', 'ある', 'いう', '自分'}
        
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

    def generate_wordcloud_image(self, text_data, file_name, width=800, height=400):
        if not text_data:
            st.warning(f"{file_name} のワードクラウド用のテキストデータが空です。")
            return None

        wc = WordCloud(
            font_path=self.wc_font_path,
            width=width,
            height=height,
            background_color="black",
            max_words=100,
            min_font_size=10,
            collocations=False
        )
        
        word_counts = Counter(text_data)
        wc.generate_from_frequencies(word_counts)

        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(file_name, fontproperties=font_prop, fontsize=18, color='white')
        
        return fig

    def draw_network_graph_matplotlib(self, G, title="共起ネットワークグラフ"):
        if G is None or G.number_of_nodes() == 0:
            st.warning(f"{title} のネットワークグラフデータがありません。ノードが不足している可能性があります。")
            return None

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        if weights:
            max_weight = max(weights)
            edge_widths = [w / max_weight * 4.5 + 0.5 for w in weights] 
        else:
            edge_widths = [1] * len(G.edges())

        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.5, edge_color='gray')

        node_sizes = [G.degree(node) * 200 + 500 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='skyblue', alpha=0.9)

        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_family=font_prop.get_name())

        ax.set_title(title, fontproperties=font_prop, fontsize=18)
        ax.axis('off')
        
        return fig


    @st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
    def analyze_data(_self, df_input, top_n_keywords):
        _self.df = df_input.copy()
        _self.question_cols = [col for col in _self.df.columns if re.match(r'Q\d+回答', col)]

        st.write("### 自然言語処理分析中...")

        # 全体分析のためのテキスト抽出と前処理
        all_combined_texts_for_overall_analysis = []
        # 各回答行のオリジナルインデックスを保持するためのリスト
        overall_original_indices = [] 
        for idx, row in _self.df.iterrows():
            combined_row_text = []
            for col_name in _self.question_cols:
                preprocessed_text = _self.preprocess_text(row[col_name])
                if preprocessed_text:
                    combined_row_text.append(preprocessed_text)
            
            if combined_row_text: # 何らかの有効なテキストがある場合のみ追加
                all_combined_texts_for_overall_analysis.append(" ".join(combined_row_text))
                overall_original_indices.append(idx) # 元の行インデックスを保存

        # 有効な回答テキストのみをフィルタリング
        valid_overall_texts_with_indices = [
            (overall_original_indices[i], text) 
            for i, text in enumerate(all_combined_texts_for_overall_analysis) 
            if text
        ]
        
        st.write(f"全体での有効回答数: {len(valid_overall_texts_with_indices)}")

        analysis_results = {}
        analysis_results['overall'] = {}
        
        overall_keyword_occurrence_counts = Counter()
        overall_word_list_for_wordcloud = [] # ワードクラウド用に全ての形態素解析された単語を保持

        # ホバー情報表示のために、キーワードと元のテキストを紐づけたDataFrameを作成
        overall_detailed_keyword_data = []

        for original_idx, text_content in valid_overall_texts_with_indices:
            morphed_words = _self.morphological_analysis(text_content)
            overall_word_list_for_wordcloud.extend(morphed_words) # ワードクラウド用

            unique_words_in_text = set(morphed_words) # 回答内のユニークなキーワード
            for word in unique_words_in_text:
                overall_keyword_occurrence_counts[word] += 1
                overall_detailed_keyword_data.append({
                    'keyword': word,
                    'original_text': _self.df.loc[original_idx, _self.question_cols].astype(str).str.cat(sep=" "), # 回答全てを連結
                    'original_row_index': original_idx
                })
                
        analysis_results['overall']['word_list'] = overall_word_list_for_wordcloud # ワードクラウド用
        analysis_results['overall']['top_words'] = overall_keyword_occurrence_counts.most_common(top_n_keywords)
        analysis_results['overall']['detailed_keywords_df'] = pd.DataFrame(overall_detailed_keyword_data)

        analysis_results['overall']['tfidf_keywords'] = _self.extract_keywords_tfidf([text for _, text in valid_overall_texts_with_indices])

        analysis_results['overall']['topics'] = _self.topic_modeling_lda([text for _, text in valid_overall_texts_with_indices])
        
        overall_sentiments = [_self.sentiment_analysis(t) for _, t in valid_overall_texts_with_indices]
        analysis_results['overall']['sentiments'] = Counter([s['sentiment'] for s in overall_sentiments])

        classifications = []
        for i, row in _self.df.iterrows():
            combined_text_parts = []
            for col in _self.question_cols:
                combined_text_parts.append(_self.preprocess_text(row[col]))
            classifications.append(_self.classify_request_type(" ".join(combined_text_parts).strip()))

        temp_df = _self.df.copy()
        temp_df['classification'] = classifications 
        analysis_results['overall']['classifications'] = Counter(classifications)
        
        _self.df = temp_df

        analysis_results['overall']['collocations'] = _self.collocation_analysis([text for _, text in valid_overall_texts_with_indices])

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
            
            # 設問ごとの有効なテキストとその元のインデックスを保持
            q_texts_with_indices = []
            for idx, row in _self.df.iterrows():
                preprocessed_text = _self.preprocess_text(row[q_col])
                if preprocessed_text:
                    q_texts_with_indices.append((idx, preprocessed_text))

            st.write(f"設問 '{q_col}' の有効回答数: {len(q_texts_with_indices)}")

            analysis_results['questions'][q_col] = {}
            
            analysis_results['questions'][q_col]['keywords'] = _self.extract_keywords_tfidf([text for _, text in q_texts_with_indices])
            
            q_sentiments = [_self.sentiment_analysis(t) for _, t in q_texts_with_indices]
            analysis_results['questions'][q_col]['sentiments'] = Counter([s['sentiment'] for s in q_sentiments])
            
            analysis_results['questions'][q_col]['collocations'] = _self.collocation_analysis([text for _, text in q_texts_with_indices])
            
            q_keyword_occurrence_counts = Counter()
            q_word_list_for_wordcloud = [] # ワードクラウド用に全ての形態素解析された単語を保持

            # 設問ごとの詳細キーワードDataFrame (ホバー情報用)
            q_detailed_keyword_data = []

            for original_idx, text_content in q_texts_with_indices:
                morphed_words = _self.morphological_analysis(text_content)
                q_word_list_for_wordcloud.extend(morphed_words) # ワードクラウド用

                unique_words_in_text = set(morphed_words) # 回答内のユニークなキーワード
                for word in unique_words_in_text: # 修正済み
                    q_keyword_occurrence_counts[word] += 1
                    q_detailed_keyword_data.append({
                        'keyword': word,
                        'original_text': _self.df.loc[original_idx, q_col], # 特定の質問の回答テキスト
                        'original_row_index': original_idx
                    })

            analysis_results['questions'][q_col]['word_list'] = q_word_list_for_wordcloud # ワードクラウド用
            analysis_results['questions'][q_col]['top_words'] = q_keyword_occurrence_counts.most_common(top_n_keywords)
            analysis_results['questions'][q_col]['detailed_keywords_df'] = pd.DataFrame(q_detailed_keyword_data)
        
        return analysis_results, _self.df


def main_app():
    st.set_page_config(layout="wide")
    st.title("アンケート自由記述分析アプリ")

    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'analyzer' not in st.session_state:
        st.session_state['analyzer'] = None
    if 'selected_overall_keyword' not in st.session_state:
        st.session_state['selected_overall_keyword'] = None
    if 'selected_overall_keyword_indices' not in st.session_state:
        st.session_state['selected_overall_keyword_indices'] = []

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.sidebar.success("ファイルが正常にアップロードされました。")
            st.sidebar.dataframe(df.head())

            top_n_keywords = st.sidebar.slider(
                "表示する頻出キーワード数",
                min_value=1,
                max_value=50,
                value=st.session_state.get('top_n_keywords', 10),
                step=1,
                key='top_n_slider'
            )
            st.session_state['top_n_keywords'] = top_n_keywords

            if st.sidebar.button("分析を実行"):
                analyzer = SurveyNLPAnalyzer(df)
                analysis_results, updated_df = analyzer.analyze_data(df, top_n_keywords) 

                st.session_state['analysis_results'] = analysis_results
                st.session_state['df'] = updated_df
                st.session_state['analyzer'] = analyzer
                
                st.session_state['selected_overall_keyword'] = None
                st.session_state['selected_overall_keyword_indices'] = []

                temp_analyzer = SurveyNLPAnalyzer(updated_df)
                question_cols_for_reset = [col for col in updated_df.columns if re.match(r'Q\d+回答', col)]

                for col_name in question_cols_for_reset:
                    if f'selected_q_keyword_{col_name}' in st.session_state:
                        del st.session_state[f'selected_q_keyword_{col_name}']
                    if f'selected_q_keyword_indices_{col_name}' in st.session_state:
                        del st.session_state[f'selected_q_keyword_indices_{col_name}']
                
                st.rerun()

        except UnicodeDecodeError:
            st.error("CSVファイルのエンコーディングがUTF-8ではありません。'shift_jis'または'cp932'を試してみてください。")
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='shift_jis')
                st.sidebar.success("ファイルが正常にアップロードされました (Shift-JIS)。")
                st.session_state['df'] = df
                st.info("Shift-JISで読み込みました。分析を開始するには再度「分析を実行」ボタンを押してください。")
            except Exception as e:
                st.error(f"Shift-JISでの読み込みも失敗しました: {e}")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
    else:
        st.info("CSVファイルをアップロードして分析を開始してください。")

    if st.session_state['analysis_results'] is not None and st.session_state['analyzer'] is not None:
        analysis_results = st.session_state['analysis_results']
        df = st.session_state['df']
        analyzer = st.session_state['analyzer']
        top_n_keywords_display = st.session_state.get('top_n_keywords', 10)

        st.header("✨ 全体分析結果")

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
                org_analysis_melted = org_analysis.reset_index().melt(id_vars='組織', var_name='分類', value_name='件数')
                fig_org = px.bar(org_analysis_melted, x='組織', y='件数', color='分類', title='組織別要望分類 (全体)')
                st.plotly_chart(fig_org, use_container_width=True)
            else:
                st.info("組織別分類データなし (全体)")

        with col2:
            st.subheader(f"頻出キーワード上位{top_n_keywords_display}件")
            overall_top_words_df = pd.DataFrame(analysis_results['overall'].get('top_words', []), columns=['キーワード', '出現回数'])
            if not overall_top_words_df.empty:
                detailed_df = analysis_results['overall']['detailed_keywords_df']
                
                hover_text_map = detailed_df.groupby('keyword')['original_text'].apply(
                    lambda x: "<br>- " + "<br>- ".join(x.unique())
                ).to_dict()

                overall_top_words_df['custom_hover_text'] = overall_top_words_df['キーワード'].map(hover_text_map).fillna("該当する回答がありません。")

                overall_top_words_df['original_indices_json'] = overall_top_words_df['キーワード'].apply(
                    lambda k: json.dumps([int(idx) for idx in detailed_df[detailed_df['keyword'] == k]['original_row_index'].unique()]) # int()に変換
                )

                fig_keywords = px.bar(overall_top_words_df, x='出現回数', y='キーワード', orientation='h', 
                                    title=f'頻出キーワード上位{top_n_keywords_display} (全体)',
                                    custom_data=['custom_hover_text', 'original_indices_json'] 
                                ) 
                
                fig_keywords.update_traces(
                    hovertemplate="<b>%{y}</b><br>出現回数: %{x}<br>" +
                                  "<b>関連する回答:</b><br>%{customdata[0]}" + 
                                  "<extra></extra>" 
                )
                
                selected_points = plotly_events(
                    fig_keywords, 
                    select_event=True, 
                    key="overall_keywords_plot" 
                )

                if selected_points and 'customdata' in selected_points[0]:
                    clicked_keyword = selected_points[0]['y'] 
                    clicked_row_indices_json = selected_points[0]['customdata'][1]
                    clicked_row_indices = json.loads(clicked_row_indices_json)
                    
                    st.session_state['selected_overall_keyword'] = clicked_keyword
                    st.session_state['selected_overall_keyword_indices'] = clicked_row_indices
                    
                    st.rerun()
                
                if st.session_state['selected_overall_keyword']:
                    st.subheader(f"選択キーワード: 『{st.session_state['selected_overall_keyword']}』 の関連回答詳細")
                    
                    selected_indices = list(set(st.session_state['selected_overall_keyword_indices']))
                    if not df.empty and selected_indices:
                        display_cols = [col for col in df.columns if re.match(r'Q\d+回答', col)]
                        if '組織' in df.columns: display_cols.append('組織')
                        if '性別' in df.columns: display_cols.append('性別')
                        display_cols = list(set(display_cols))

                        st.dataframe(df.loc[selected_indices, display_cols])
                    else:
                        st.info("選択されたキーワードに関連する詳細データが見つかりませんでした。")
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
            
            st.subheader("ワードクラウド (全体)")
            overall_wc_image_fig = analyzer.generate_wordcloud_image(
                analysis_results['overall'].get('word_list', []), 
                'ワードクラウド (全体)', width=1000, height=500
            )
            if overall_wc_image_fig:
                st.pyplot(overall_wc_image_fig)
            else:
                st.info("ワードクラウドデータなし (全体)")
            
            st.subheader("共起ネットワークグラフ (全体)")
            overall_collocations = analysis_results['overall'].get('collocations', [])
            overall_network_graph = analyzer.create_network_graph(overall_collocations)
            
            if overall_network_graph and overall_network_graph.number_of_nodes() > 0:
                fig_overall_network = analyzer.draw_network_graph_matplotlib(overall_network_graph, "共起ネットワークグラフ (全体)")
                if fig_overall_network:
                    st.pyplot(fig_overall_network)
            else:
                st.info("共起ネットワークグラフのデータなし (全体)。共起語が見つからないか、ノードが作成できませんでした。")


        st.markdown("---")

        st.header("📝 設問別分析結果")

        if analyzer.question_cols: 
            for q_col in analyzer.question_cols:
                if f'selected_q_keyword_{q_col}' not in st.session_state:
                    st.session_state[f'selected_q_keyword_{q_col}'] = None
                if f'selected_q_keyword_indices_{q_col}' not in st.session_state:
                    st.session_state[f'selected_q_keyword_indices_{q_col}'] = []

                st.subheader(f"### {q_col} の分析結果")
                q_results = analysis_results['questions'][q_col]

                col_q1, col_q2, col_q3 = st.columns(3)

                with col_q1:
                    st.write(f"#### 頻出キーワード上位{top_n_keywords_display}件")
                    q_top_words_df = pd.DataFrame(q_results.get('top_words', []), columns=['キーワード', '出現回数'])
                    if not q_top_words_df.empty:
                        detailed_df = q_results['detailed_keywords_df']
                        
                        hover_text_map_q = detailed_df.groupby('keyword')['original_text'].apply(
                            lambda x: "<br>- " + "<br>- ".join(x.unique())
                        ).to_dict()

                        q_top_words_df['custom_hover_text'] = q_top_words_df['キーワード'].map(hover_text_map_q).fillna("該当する回答がありません。")
                        
                        q_top_words_df['original_indices_json'] = q_top_words_df['キーワード'].apply(
                            lambda k: json.dumps([int(idx) for idx in detailed_df[detailed_df['keyword'] == k]['original_row_index'].unique()]) # int()に変換
                        )

                        fig_q_keywords = px.bar(q_top_words_df, x='出現回数', y='キーワード', orientation='h',
                                                title=f'頻出キーワード上位{top_n_keywords_display} ({q_col})',
                                                custom_data=['custom_hover_text', 'original_indices_json'] 
                                            ) 

                        fig_q_keywords.update_traces(
                            hovertemplate="<b>%{y}</b><br>出現回数: %{x}<br>" +
                                          "<b>関連する回答:</b><br>%{customdata[0]}" + 
                                          "<extra></extra>"
                        )
                        
                        selected_points_q = plotly_events(
                            fig_q_keywords, 
                            select_event=True, 
                            key=f"q_{q_col}_keywords_plot" 
                        )

                        if selected_points_q and 'customdata' in selected_points_q[0]:
                            clicked_keyword_q = selected_points_q[0]['y'] 
                            clicked_row_indices_json_q = selected_points_q[0]['customdata'][1]
                            clicked_row_indices_q = json.loads(clicked_row_indices_json_q)
                            
                            st.session_state[f'selected_q_keyword_{q_col}'] = clicked_keyword_q
                            st.session_state[f'selected_q_keyword_indices_{q_col}'] = clicked_row_indices_q
                            
                            st.rerun()
                        
                        if st.session_state.get(f'selected_q_keyword_{q_col}'):
                            with st.expander(f"選択キーワード: 『{st.session_state[f'selected_q_keyword_{q_col}']}』 の関連回答詳細"):
                                selected_indices_q = list(set(st.session_state[f'selected_q_keyword_indices_{q_col}']))
                                if not df.empty and selected_indices_q:
                                    display_cols_q = [q_col]
                                    if '組織' in df.columns: display_cols_q.append('組織')
                                    if '性別' in df.columns: display_cols_q.append('性別')
                                    display_cols_q = list(set(display_cols_q))
                                    
                                    st.dataframe(df.loc[selected_indices_q, display_cols_q])
                                else:
                                    st.info("選択されたキーワードに関連する詳細データが見つかりませんでした。")
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

                with col_q3:
                    st.write("#### ワードクラウド")
                    q_wc_image_fig = analyzer.generate_wordcloud_image(
                        q_results.get('word_list', []), 
                        f'ワードクラウド ({q_col})', width=800, height=400
                    )
                    if q_wc_image_fig:
                        st.pyplot(q_wc_image_fig)
                    else:
                        st.info(f"ワードクラウドデータなし ({q_col})")
                    
                    st.write("#### 共起ネットワークグラフ")
                    q_collocations = q_results.get('collocations', [])
                    q_network_graph = analyzer.create_network_graph(q_collocations)
                    
                    if q_network_graph and q_network_graph.number_of_nodes() > 0:
                        fig_q_network = analyzer.draw_network_graph_matplotlib(q_network_graph, f"共起ネットワークグラフ ({q_col})")
                        if fig_q_network:
                            st.pyplot(fig_q_network)
                    else:
                        st.info(f"共起ネットワークグラフのデータなし ({q_col})。共起語が見つからないか、ノードが作成できませんでした。")


                st.markdown("---") 
        else:
            st.info("分析対象の設問回答列（Q〇〇回答）が見つかりませんでした。CSVファイルの列名をご確認ください。")


if __name__ == '__main__':
    main_app()