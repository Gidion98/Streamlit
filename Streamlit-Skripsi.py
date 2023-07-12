import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import numpy as np
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Menambahkan judul aplikasi
st.write("# Sentiment Analysis")

# Menampilkan data dalam bentuk tabel
st.write('### DataSet' )
total_data = pd.read_csv('Copy of DataSet1.csv', encoding='ISO-8859-1')
data = total_data.head(10)
st.dataframe(data)


# Merubah data menjadi lower case
st.write('### Lower Case' )
import re
review = total_data.columns.values[1]
sentiment = total_data.columns.values[2]
def process_review(review):
    if isinstance(review, str):
        review = review.lower()
        review = re.sub('@[^\s]+', '', review)
        review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', review)
        review = re.sub(r"\d+", " ", review)
        review = re.sub('&quot;', " ", review)
        review = re.sub(r"\b[a-zA-Z]\b", "", review)
        review = re.sub(r"[^\w\s]", " ", review)
        review = re.sub(r'(.)\1+', r'\1\1', review)
        review = re.sub(r"\s+", " ", review)
    else:
        review = np.nan  # or any other desired value for missing reviews
    return review
total_data['processed_review'] = np.vectorize(process_review)(total_data[review])
st.dataframe(total_data.head(10))


# Menampilkan data stemmer
st.write('### Stemming' )
stemmer = PorterStemmer()
total_data = total_data.apply(lambda x: [stemmer.stem(i) if isinstance(i, str) else i for i in x])
st.dataframe(total_data.head(10))


# Menampilkan data stopword removing
st.write('### Stopword Removing' )
stopwords_english = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords_english]
    return ' '.join(filtered_tokens)
total_data['processed_review'] = total_data['processed_review'].apply(remove_stopwords)
st.dataframe(total_data.head(10))


# Membagi data menjadi 3 kelas
st.write('### Membagi Data Menjadi 3 Kelas' )
from textblob import TextBlob
review = total_data.columns.values[1] #content
sentiment = total_data.columns.values[2] #score
def sentiment_analysis(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'
total_data['sentiment'] = ''
total_data['sentiment'] = total_data['processed_review'].apply(sentiment_analysis)
st.dataframe(total_data.head(10))


# Menampilkan data tokenized
st.write('### Tokenized' )
tokenized_review = total_data['processed_review'].apply(lambda x: x.split())
st.dataframe(tokenized_review.head(10))


# Menampilkan pie chart untuk persentase sentimen
st.write('### Sentiment Distribution' )
sentiment_counts = total_data['sentiment'].value_counts()
sentiments = sentiment_counts.index.tolist()
slices = sentiment_counts.values.tolist()
colors = ['g', 'r', 'b']

fig, ax = plt.subplots()
ax.pie(slices, labels=sentiments, colors=colors, startangle=90, shadow=True,
       explode=(0, 0.1, 0), autopct='%1.2f%%')
ax.legend()
st.pyplot(fig)


# Menampilkan word cloud untuk kata-kata positif
st.write('### Word Cloud - Positive Words' )
st.image('positive_words.png')


# Menampilkan word cloud untuk kata-kata negatif
st.write('### Word Cloud - Negative Words' )
st.image('negetive_words.png')


# Menampilkan Tf-Idf
st.write('### Menampilkan Tf-Idf' )
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3))
final_vectorized_data = tf_idf_vectorizer.fit_transform(total_data['processed_review'])
feature_names = tf_idf_vectorizer.get_feature_names_out()
for doc_idx, doc in enumerate(final_vectorized_data):
    st.write("Document ", doc_idx + 1)
    count = 0
    for feature_idx, tfidf_score in zip(doc.indices, doc.data):
        st.write(feature_names[feature_idx], ": ", tfidf_score)
        count += 1
        if count >= 30:
            break
    if count >= 30:
        break


# Membagi data menjadi data train dan data test
# st.write('### Bagi Data Menjadi Data Train dan Data Test')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_vectorized_data, total_data['sentiment'],
                                                    test_size=0.2, random_state=69)

# st.write("X-Train : ",X_train.shape)
# st.write("X-Test : ",X_test.shape)
# st.write("Y-Train : ",y_train.shape)
# st.write("Y-Test : ",y_test.shape)


# Jalankan Program Naive Bayes
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier

model_naive = MultinomialNB().fit(X_train, y_train)
predicted_naive = model_naive.predict(X_test)


# Menampilkan confusion matrix
st.write('### Confusion Matrix' )
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(dpi=600)
mat = confusion_matrix(y_test, predicted_naive)
sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

plt.title('Confusion Matrix')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

fig = plt.gcf()
fig.set_size_inches(10, 8)
st.pyplot(fig)


# Menampilkan Akurasi, Precision, Recall, F-Measure
st.write('### Cek Tingkat Akurasi, Precision, Recall, F-Measure')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

score_naive = accuracy_score(predicted_naive, y_test)
precision_naive = precision_score(y_test, predicted_naive, average='weighted')
recall_naive = recall_score(y_test, predicted_naive, average='weighted')
f1_score_naive = f1_score(y_test, predicted_naive, average='weighted')

st.write("Accuracy : ", score_naive)
st.write("Precision : ", precision_naive)
st.write("Recall : ", recall_naive)
st.write("F-measure : ", f1_score_naive)


# Menampilkan Classification Report
from sklearn.metrics import classification_report
st.write('### Classification Report')
st.text(classification_report(y_test, predicted_naive))


# Representasi vector kata / mencari kata yang mirip
st.write('### Mencari Kata Yang Mirip')
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
word2vec_model = Word2Vec(tokenized_review, vector_size=200, window=5, min_count=2, sg=1, workers=4)
word2vec_model.train(tokenized_review, total_examples=len(tokenized_review), epochs=10)
st.write('Kata Yang Mirip Dengan "Good" :')
st.dataframe(word2vec_model.wv.most_similar('good'))
st.write('Kata Yang Mirip Dengan "Bad" :')
st.dataframe(word2vec_model.wv.most_similar('bad'))


# Preprocessing Topik Modeling
st.write('### Preprocessing Topik Modeling ')
st.text('Document')
document = []

for i in range(len(tokenized_review)):
    if len(tokenized_review.iloc[i]) >= 4:
        a=tokenized_review.iloc[i][3]
        document.append(a)
st.dataframe(document[0:10])

st.text('Doc_Clean')
doc_clean = tokenized_review
st.dataframe(doc_clean[0:10])


# Proses Topik Modeling
import gensim
from gensim import corpora

dictionary = corpora.Dictionary(doc_clean)
st.write('### Proses Topik Modeling')
st.text('Dictionary')
st.write(dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Membuat object untuk LDA model menggunakan gensim library
Lda = gensim.models.ldamodel.LdaModel

total_topics = 3 # jumlah topik yang akan di extract
number_words = 10 # jumlah kata per topik

# Jalankan dan Uji LDA model pada document term matrix.
lda_model = Lda(doc_term_matrix, num_topics=total_topics, id2word = dictionary, passes=50)
st.text('Pengujian LDA Model Pada Document Term Matrix')
st.dataframe(lda_model.show_topics(num_topics=total_topics, num_words=number_words))

# Word Count of Topic Keywords
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in doc_clean for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df_imp_wcount = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
st.text('Topik Word Count')
st.dataframe(df_imp_wcount)

#Dominant topic and its percentage contribution in each topic
def format_topics_sentences(ldamodel=None, corpus=doc_term_matrix, texts=document):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=doc_term_matrix, texts=doc_clean)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

st.text('Dominant Topik')
st.dataframe(df_dominant_topic.head(10))

import pickle
import pyLDAvis
import pyLDAvis.gensim
# Visualize the topics
#pyLDAvis.enable_notebook()

import os
LDAvis_data_filepath = os.path.join('ldavis_prepared_'+str(total_topics))

corpus = [dictionary.doc2bow(text) for text in doc_clean]

# proses ini mungkin agak lama
import pyLDAvis.gensim

# Memeriksa apakah LDAvis data sudah ada
if not os.path.exists(LDAvis_data_filepath):
    # Proses ini mungkin memakan waktu lama
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
else:
    # Memuat data LDAvis yang sudah ada
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

# Menyimpan visualisasi dalam format HTML
html_filepath = 'ldavis_prepared_' + str(total_topics) + '.html'
pyLDAvis.save_html(LDAvis_prepared, html_filepath)

# Menampilkan visualisasi di Streamlit
st.write('### Visualisasi LDA')
st.components.v1.html(open(html_filepath, 'r', encoding='utf-8').read(), width=900, height=800)
