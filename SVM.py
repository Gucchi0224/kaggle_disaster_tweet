import os, re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# 形態素解析
def get_tokens(df):
    # stopwordsの定義
    tokens = []
    raw_tokens = []
    pos_tag_list = ['JJ','NN','RB','VB']
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    stop_words = stopwords.words('english')
    for sentence in df:
        # 単語の取得
        raw_token = []
        raw_token_ = []
        for word in sentence.split():
            raw_token_.append(word)
            # 単語を原型に変換
            word = wn.morphy(word)
            if word != None:
                raw_token.append(word)
                
        # 原文の単語をそのまま格納
        raw_tokens.append(raw_token_)
        # 品詞の特定
        token = []
        pos = nltk.pos_tag(raw_token)
        for word in pos:
            if word[0] not in stop_words:
                if word[1] in pos_tag_list:
                    token.append(word[0])

        # 1つの英文(stopwordを除く)で得られたtokenを格納
        tokens.append(token)

    return tokens, raw_tokens

# BOW
def make_bow(tokens):
    # 辞書の作成
    token_list = []
    for token in tokens:
        token_list += token
        
    dic = list(set(token_list))
    # BOWの作成
    bow = []
    for token in tokens:
        zero_list = np.zeros(len(dic))
        for word in token:
            zero_list[dic.index(word)] += 1

        bow.append(zero_list)

    return bow

# TF-IDF
def calc_tfidf(tokens):
    # 形態素解析で残ったwordのみで文章を作成
    sentences = []
    for token in tokens:
        sentence = ""
        for word in token:
            sentence += " " + word

        sentences.append(sentence)
    
    # tf-idfの計算
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(sentences)
    df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    
    return df_tfidf

if __name__ == "__main__":
    # フォルダの作成
    csv_path = './predict_csv'
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    
    dir_path = "./figure"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    # データの読み込み
    data_path = './nlp-getting-started/'
    train_df = pd.read_csv(data_path + 'train.csv')
    test_df = pd.read_csv(data_path + 'test.csv')

    # テキスト文と結果の格納
    y_train = train_df['target']

    # train_df, test_dfのtext部分だけを取り出して結合
    text_df = pd.concat([train_df['text'], test_df['text']], axis=0, ignore_index=True)

    # 要らない文字列を削除
    sentences = []
    del_index = []
    for i, text in enumerate(text_df):
        text = re.sub('RT ', '', text)
        text = text.lower()
        text = re.sub('[^0-9a-z ]+', '', text)
        text = re.sub('http[a-z\S]+', '', text)
        text_df[i] = text
        if i < len(train_df):
            if text not in sentences:
                sentences.append(text)
            else:
                del_index.append(i)
            
    # 同じ文章を削除, テキスト文を保存
    text_df1 = text_df.drop(index=del_index)
    y_train1 = y_train.drop(index=del_index)
    text_df.to_csv("nlp-getting-started\\text.csv")
    text_df1.to_csv("nlp-getting-started\\text(del_common_string).csv")
    
    # 形態素解析
    tokens, raw_tokens = get_tokens(text_df1)
    # BOW or TF-IDF
    #bow = make_bow(raw_tokens)
    tfidf = calc_tfidf(tokens)
    
    # Numpy配列に変換
    X = np.array(tfidf)
    X_train, X_test = X[:-len(test_df)], X[-len(test_df):]
    y_train = np.array(y_train1)

    # SVMモデルのインスタンス化
    model = LinearSVC()
    
    # 学習
    print("="*50)
    print('start training...')
    model.fit(X_train, y_train)
    print('Training accuracy:', model.score(X_train, y_train))
    
    # テスト
    y_pred = model.predict(X_test)
    test_df['target'] = y_pred
    pred_df = test_df[['id', 'target']].set_index('id')
    pred_df.to_csv("predict_csv\\prediction_data_by_SVM.csv", index_label=["id"])
