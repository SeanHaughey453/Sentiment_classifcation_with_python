import json
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant

def main():
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    corpus = []
    labels = []

    #with open('new_dataset.json', 'r') as f:
    # with open('yelp_academic_dataset_review.json', 'r') as f:
    #     data_list = json.load(f)
    # data = pd.DataFrame(data_list)

    #data['label'] = data['stars'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')

    # for i, row in data.iterrows():
    #     text = re.sub(r'[^a-zA-Z]', ' ', row['text'])
    #     text = text.lower().split()
    #     text = [word for word in text if word not in stopwords.words('english')]
    #     text = [lemmatizer.lemmatize(word) for word in text]
    #     corpus.append(text)
    #     labels.append(row['label'])

    with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5000:  # Limit to the first 1000 entries
                review = json.loads(line)
                text = re.sub(r'[^a-zA-Z]', ' ', review['text'])
                text = text.lower().split()
                text = [word for word in text if word not in stopwords.words('english')]
                text = [lemmatizer.lemmatize(word) for word in text]
                corpus.append(text)
                label = 'positive' if review['stars'] > 3 else 'negative' if review['stars'] < 3 else 'neutral'
                labels.append(label)
            else:
                break

    word2vec_model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
    vocabulary = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
    max_index = max(vocabulary.values())
    input_dim = max_index + 1

    embedding_dim = 100
    embedding_matrix = np.zeros((input_dim, embedding_dim))
    for word, idx in vocabulary.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]

    sequences = [[vocabulary[word] for word in text if word in vocabulary] for text in corpus]
    sequences_padded = pad_sequences(sequences, maxlen=100, padding='post')

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_categorical, test_size=0.2, random_state=42)

    # model = Sequential()
    # model.add(Embedding(input_dim=input_dim,output_dim=embedding_dim,embeddings_initializer=Constant(embedding_matrix),input_length=100,trainable=False))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(GlobalMaxPooling1D())
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(3, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model = Sequential()
    model.add(Embedding(input_dim=input_dim,output_dim=embedding_dim,embeddings_initializer=Constant(embedding_matrix),input_length=100,trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform([np.argmax(y) for y in y_pred])
    y_true_labels = label_encoder.inverse_transform([np.argmax(y) for y in y_test])
    
    print(classification_report(y_true_labels, y_pred_labels))

if __name__ == '__main__':
    main()
