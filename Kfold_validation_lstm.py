import numpy as np
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Bidirectional, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

def main():
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    corpus = []
    labels = []

    with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5000:
                review = json.loads(line)
                text = re.sub(r'[^a-zA-Z]', ' ', review['text'])
                text = text.lower().split()
                text = [word for word in text if word not in stop_words]
                text = [lemmatizer.lemmatize(word) for word in text]
                corpus.append(text)
                label = 'positive' if review['stars'] > 3 else 'negative' if review['stars'] < 3 else 'neutral'
                labels.append(label)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    word2vec_model = Word2Vec(corpus, vector_size=100, window=5, min_count=2, workers=4)
    vocabulary = {word: idx + 1 for idx, word in enumerate(word2vec_model.wv.index_to_key)}
    sequences = [[vocabulary[word] for word in text if word in vocabulary] for text in corpus]
    sequences_padded = pad_sequences(sequences, maxlen=100, padding='post')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    all_classification_reports = []  
    for train, test in kfold.split(sequences_padded, labels_categorical):

        inputs = Input(shape=(100,))
        embedding = Embedding(input_dim=len(vocabulary) + 1, output_dim=100, trainable=True)(inputs)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(embedding)
        dropout1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(64, return_sequences=False))(dropout1)
        dropout2 = Dropout(0.5)(lstm2)
        dense = Dense(64, activation='tanh')(dropout2)
        dropout3 = Dropout(0.5)(dense)
        outputs = Dense(3, activation='softmax')(dropout3)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        X_train, X_test = sequences_padded[train], sequences_padded[test]
        y_train, y_test = labels_categorical[train], labels_categorical[test]
        
        model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))
        
        y_pred = model.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform([np.argmax(y) for y in y_pred])# this seperates the pridcated labels form y pred
        y_true_labels = label_encoder.inverse_transform([np.argmax(y) for y in y_test])# this seperates the true labels from y_tes
        #both of these are seperated to be put into the classfication report

        
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        all_classification_reports.append(report)

        print(f'Fold {fold_no}')
        fold_no += 1

    for i, report in enumerate(all_classification_reports, 1):
        print(f'Classification Report for Fold {i}')
        print(report)

if __name__ == '__main__':
    main()
