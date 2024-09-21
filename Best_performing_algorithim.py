import numpy as np
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
            else:
                break
    
    # Encoding labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    # Word2Vec model
    word2vec_model = Word2Vec(corpus, vector_size=100, window=5, min_count=2, workers=4)
    vocabulary = {word: idx + 1 for idx, word in enumerate(word2vec_model.wv.index_to_key)}

    # Create sequences for LSTM input
    sequences = [[vocabulary[word] for word in text if word in vocabulary] for text in corpus]
    sequences_padded = pad_sequences(sequences, maxlen=100, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_categorical, test_size=0.2, random_state=42)# change to .4 for 60/40 split

    inputs1 = Input(shape=(100,))
    embedding_layer = Embedding(input_dim=len(vocabulary) + 1, output_dim=100, trainable=True)(inputs1)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    dropout1 = Dropout(0.5)(lstm1)
    lstm2 = Bidirectional(LSTM(64, return_sequences=False))(dropout1)
    dropout2 = Dropout(0.5)(lstm2)
    dense1 = Dense(64, activation='tanh')(dropout2)
    dropout3 = Dropout(0.5)(dense1)
    output_layer = Dense(3, activation='softmax')(dropout3)
    model = Model(inputs=inputs1, outputs=output_layer)

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform([np.argmax(y) for y in y_pred])
    y_true_labels = label_encoder.inverse_transform([np.argmax(y) for y in y_test])
    
    print('Classfication Report')
    print(classification_report(y_true_labels, y_pred_labels))

    print('Confusion Matrix')
    print(confusion_matrix(y_true_labels, y_pred_labels))


if __name__ == '__main__':
    main()
