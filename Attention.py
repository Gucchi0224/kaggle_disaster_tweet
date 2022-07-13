import os, re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten
from keras.layers import Input, Reshape, Bidirectional
from keras_self_attention import SeqSelfAttention
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

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

# 損失の可視化
def plot_history_loss(history):
    axL.plot(history.history['loss'], label="loss for training")
    axL.plot(history.history['val_loss'], label="loss for validation")
    axL.set_title('model_loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# 正解率の可視化
def plot_history_acc(history):
    axR.plot(history.history['accuracy'], label="acc for training")
    axR.plot(history.history['val_accuracy'], label="acc for validation")
    axR.set_title('model_accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

if __name__ == "__main__":
    # フォルダの作成
    csv_path = './predict_csv'
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    
    dir_path = "./loss-acc"
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
    tokens, raw_tokens = get_tokens(text_df)
    # BOW
    #bow = make_bow(raw_tokens)
    # TF-IDF
    tfidf = calc_tfidf(raw_tokens)
    
    # Numpy配列に変換
    X = np.array(tfidf)
    X_train, X_test = X[:-len(test_df)], X[-len(test_df):]
    y_train = np.array(y_train)
    
    # 訓練と評価で分割
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2)

    # モデル構築
    inputs = Input(shape=(X_train.shape[1],))
    x = Reshape((1, X_train.shape[1]))(inputs)
    x = SimpleRNN(units=20, return_sequences=True)(x)
    x = SeqSelfAttention(units=64, attention_activation='tanh')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    loss_weight = [0.9]
    model.compile(loss='binary_crossentropy', loss_weights=loss_weight, optimizer='adam', metrics=['accuracy'])

    # 学習
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_val, y_val), callbacks=[callback])

    # 学習の様子を可視化
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    plot_history_loss(history)
    plot_history_acc(history)
    plt.savefig("loss-acc\\loss-acc(Attention).jpg", bbox_inches='tight')
    plt.close()
    
    # テスト
    y_pred = model.predict(X_test, batch_size=32)
    y_pred = np.round(y_pred).astype(int)
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1]))
    test_df['target'] = y_pred
    pred_df = test_df[['id', 'target']].set_index('id')
    pred_df.to_csv("predict_csv\\prediction_data_by_Attention.csv", index_label=["id"])
