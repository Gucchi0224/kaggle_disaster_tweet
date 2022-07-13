import os, re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten
from keras.layers import Input, Bidirectional, Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import regularizers
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

        # 1つの英文(stopword, pos_tag_list以外の品詞を除く)で得られたtokenを格納
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
            sentence += word + " "

        sentences.append(sentence)
    
    # tf-idfの計算
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(sentences)
    df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    
    return df_tfidf

# 損失の可視化
def plot_history_loss(train_loss_list, val_loss_list):
    axL.plot(train_loss_list, label="loss for training")
    axL.plot(val_loss_list, label="loss for validation")
    axL.set_title('model_loss')
    axL.set_xlabel('epochs')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# 正解率の可視化
def plot_history_acc(train_acc_list, val_acc_list):
    axR.plot(train_acc_list, label="acc for training")
    axR.plot(val_acc_list, label="acc for validation")
    axR.set_title('model_accuracy')
    axR.set_xlabel('epochs')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

# batchごとにloss, accの平均を求めてグラフ化
def loss_acc_batch_average(historys):
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for history_batch in historys:
        train_loss_batch_list = []
        val_loss_batch_list = []
        train_acc_batch_list = []
        val_acc_batch_list = []
        for history in history_batch:
            train_loss_batch_list.append(history.history['loss'][0])
            val_loss_batch_list.append(history.history['val_loss'][0])
            train_acc_batch_list.append(history.history['accuracy'][0])
            val_acc_batch_list.append(history.history['val_accuracy'][0])
            
        train_loss_list.append(np.sum(np.array(train_loss_batch_list))/len(train_loss_batch_list))
        val_loss_list.append(np.sum(np.array(val_loss_batch_list))/len(val_loss_batch_list))
        train_acc_list.append(np.sum(np.array(train_acc_batch_list))/len(train_acc_batch_list))
        val_acc_list.append(np.sum(np.array(val_acc_batch_list))/len(val_acc_batch_list))
    
    # 可視化
    plot_history_loss(train_loss_list, val_loss_list)
    plot_history_acc(train_acc_list, val_acc_list)

if __name__ == "__main__":
    # フォルダの作成
    csv_path = './predict_csv'
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    
    fig_path = "./loss-acc"
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    
    result_path = "./result_val"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    # データの読み込み
    data_path = './nlp-getting-started/'
    train_df = pd.read_csv(data_path + 'train.csv')
    test_df = pd.read_csv(data_path + 'test.csv')

    # テキスト文と結果の格納
    y_train = train_df['target']

    # train_df, test_dfのtext部分だけを取り出して結合
    text_df = pd.concat([train_df['text'], test_df['text']], axis=0, ignore_index=True)

    # 要らない文字列を削除
    del_index = []
    for i, text in enumerate(text_df):
        text = re.sub('RT ', '', text)
        text = text.lower()
        text = re.sub('[^0-9a-z ]+', '', text)
        #text = re.sub('[0-9]+', '0', text)
        text = re.sub('http[a-z\S]+', '', text)
        text_df[i] = text
        if i < len(train_df):
            if text == '':
                del_index.append(i)
            
    # 同じ文章を削除, テキスト文を保存
    text_df = text_df.drop(index=del_index)
    text_df.to_csv("nlp-getting-started\\text.csv")
    
    # 形態素解析
    tokens, raw_tokens = get_tokens(text_df)
    # BOW or TF-IDF
    #bow = make_bow(raw_tokens)
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
    x = SimpleRNN(units=64, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    outputs = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    loss_weight = [0.9]
    model.compile(loss='binary_crossentropy', loss_weights=loss_weight, optimizer='rmsprop', metrics=['accuracy'])

    # 学習
    historys = []
    flag = 0
    val_loss = 1.0
    stop_fit_count = 0
    MAX_EPOCHS = 30
    BATCH_SIZE = 256
    for epoch in range(MAX_EPOCHS):
        history_batch = []
        print('Epoch %s/%s' % (epoch+1, MAX_EPOCHS))
        # 学習データ, 評価データをランダムにシャッフル
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]
        idx = np.random.permutation(X_val.shape[0])
        X_val = X_val[idx]
        y_val = y_val[idx]
        # バッチサイズごとにデータを分ける
        X_split_train = [X_train[i:i+BATCH_SIZE] for i in range(0, X_train.shape[0], BATCH_SIZE)]
        y_split_train = [y_train[i:i+BATCH_SIZE] for i in range(0, y_train.shape[0], BATCH_SIZE)]
        X_split_val = [X_val[i:i+BATCH_SIZE] for i in range(0, X_val.shape[0], BATCH_SIZE)]
        y_split_val = [y_val[i:i+BATCH_SIZE] for i in range(0, y_val.shape[0], BATCH_SIZE)]
        # バッチサイズごとに学習
        for (X_batch_train, y_batch_train, X_batch_val, y_batch_val) in zip(X_split_train, y_split_train, X_split_val, y_split_val):
            history = model.fit(X_batch_train, y_batch_train, validation_data=(X_batch_val, y_batch_val))
            history_batch.append(history)
            # val_lossの最小値よりも小さい値がstop_fit_count回続いたら学習終了
            if history.history['val_loss'][0] < val_loss:
                val_loss = history.history['val_loss'][0]
                stop_fit_count = 0
            else:
                stop_fit_count += 1
                if stop_fit_count == 15:
                    flag = 1
                    break
                
        # epochごとに履歴を保存
        historys.append(history_batch)
        if flag == 1:
            break
    
    # 検証データで分類正解率を可視化する
    y_pred = model.predict(X_val, batch_size=16)
    y_pred = np.round(y_pred)
    plt.rcParams["figure.figsize"]=(5,5)
    cm = confusion_matrix(y_val, y_pred, normalize='true')
    cmp = ConfusionMatrixDisplay(cm, display_labels=["fake","real"])
    cmp.plot(cmap=plt.cm.Blues, xticks_rotation=0)
    plt.title("real_or_fake_accident")
    plt.savefig("result_val\\RNN-batch.jpg", bbox_inches='tight')
    plt.close()
    
    # 学習の様子を可視化
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    loss_acc_batch_average(historys)
    plt.savefig("loss-acc\\RNN-batch.jpg", bbox_inches='tight')
    plt.close()
    
    # テスト
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.round(y_pred).astype(int)
    test_df['target'] = y_pred
    pred_df = test_df[['id', 'target']].set_index('id')
    pred_df.to_csv("predict_csv\\prediction_data_SimpleRNN_batch.csv", index_label=["id"])
    