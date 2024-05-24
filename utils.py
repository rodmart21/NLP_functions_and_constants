# Some functions to clean the data in NLP projects.
# We have in other .py all the neccesary constants we are using.

import re
import spacy
from nltk.corpus import stopwords
from src.constants import * # cargamos las constantes
#from src.clases import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from scikitplot.metrics import plot_confusion_matrix 
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import normalize
import joblib
import sys

# Eliminate characters in a foreign language. In this case we eliminate characters in catalan.
def elim_cat(df, columna, pat_cat):
    df['DELETE'] = df[columna].apply(lambda x: pat_cat.search(x))
    df['DELETE'] = df['DELETE'].apply(lambda x: True if x is None else False) 
    df = df[df['DELETE'] == True] 
    df = df.drop(columns=['DELETE'])
    return df


# Depending on the use case, people may sometimes repeat some letters just to finish the blank space
# that they have to write. This four functions help us to identify this cases and solve them.
def ratio(word):
    letters = list(word)
    nu_letters = len(letters)
    unique = set(letters)
    nu_unique = len(unique)
    ratio = nu_letters / nu_unique
    if ratio <= 2:
        return True
    return False

def rep_let(word,letters_allow_rep,pat):
    if set(re.findall(pat,word))-letters_allow_rep:
        return False
    return True

def conson_rep(word, pat_rep_cons):
    if re.search(pat_rep_cons,word) is not None:
        return False
    return True

def validacion(words,letters_allow_rep,pat,pat_cons_rep):
    words_list = words.split()
    lista = []
    for word in words_list:
        if ratio(word) and rep_let(word, letters_allow_rep, pat) and conson_rep(word, pat_cons_rep):
            lista.append(word)
    new_cad = " ".join(lista)
    return new_cad


# In some cases, users are just filling in the blanks by repeating the same word or group of words.
def eliminar_rep_palabras1(frase, pat_pal):
    frase_original = frase  
    while True:
        frase = re.sub(pat_pal, r"\1", frase, flags=re.I)  
        if frase == frase_original: 
            break 
        frase_original = frase  
    return frase


# Function to eliminate the accent.
def remove_tildes(word,dict_sub_accent):
    for k, v in dict_sub_accent.items():
        word = re.sub(k, v, word)
    return word


# Functions to eliminate the repeated vowels.
def vow_rep(word, pat_vow):
    if re.search(pat_vow,word) is not None:
        return False
    return True

def validat_vow(words, pat_vow):
    words_list = words.split()
    list=[]
    for word in words_list:
        if vow_rep(word, pat_vow):
            list.append(word)
    new_cad=" ".join(list)
    return new_cad


# Eliminate just in case there is only one word.
def long_words(row):
    words = row.split()
    if len(words) > 1:  
        return True
    return False


# Function we use to apply stemming, lematize or stopwords elimination.
def apply_preprocessing(text: str,
                        rm_stopwords: bool,
                        stem: bool,
                        lemmatize: bool,
                        STEMMER_ES=None,
                        SPACY_NLP_ES=None,
                        stopwords_no_tilde=None) -> str:
    words_list = text.split()
    if rm_stopwords and stopwords_no_tilde:
        words_list = [word for word in words_list if word.lower() not in stopwords_no_tilde]
    if stem and STEMMER_ES:
        words_list = [STEMMER_ES.stem(word) for word in words_list]
    if lemmatize and SPACY_NLP_ES:
        doc_spacy = SPACY_NLP_ES(" ".join(words_list))
        words_list = [word.lemma_ for word in doc_spacy]
    text = " ".join(words_list)
    return text


# It creates a dictionary where each word in the sentence is a key, 
# and the value associated with each key is the count of how many times that word appears in the sentence. 
# Finally, it returns this dictionary.
def dicc_pal(frase):
    list_words = frase.split()
    dict_palabras = {}
    for word in list_words:
        dict_palabras[word] = list_words.count(word)
    return dict_palabras


# The number of records in each category is different, and there are significant differentials 
# between them. Therefore, it is necessary to create a function capable of compensating 
# for this difference so that the model does not exhibit bias towards certain categories.
def balancing(cat_1, cat_2, cat_3, cert, no_comp):
    # Sorting the values in descending order:
    sorted_values = sorted([cat_1, cat_2, cat_3, cert, no_comp], reverse=True)
    max_1, max_2, max_3, max_4, mini = sorted_values
    total_sum = sum(sorted_values)

    # Calculating the ratio between the maximum and minimum value:
    rel = max_1 / mini
    proportions = [val / total_sum for val in [max_1, max_2, max_3, max_4, mini]]

    # Determining the new values based on the relationship:
    if rel > 10:
        new_sum = total_sum / (rel / 3) + mini
    elif 5 < rel < 10:
        new_sum = total_sum / (rel / 2) + mini
    else:
        return max_1.round(), max_2.round(), max_3.round(), max_4.round(), mini

    new_rel = mini / new_sum
    new_values = [prop * new_sum if prop > new_rel else prop * new_sum + mini for prop in proportions]

    return tuple(map(round, new_values)), mini


# A function similar to this one can be used to make a manual categorization based on some terms that
# we will find using RegEx.
def create_cat_1_level(dictionary):
    cat = "Unknown"
    if re.search(r'\b(...)\w*\b', ' '.join(dictionary.keys()), re.IGNORECASE) and re.search(r'\b(...)\w*\b', ' '.join(dictionary.keys()), re.IGNORECASE):
        cat = "..."
    elif re.search(r'\b(...)\w*\b', ' '.join(dictionary.keys()), re.IGNORECASE):
        cat = "..."
    return cat


# Function to split in train/test excluding some category that we can choose.
def tr_te_split(df_cat,columna,categoria_excl,percent):
    df_cat_new = df_cat[df_cat[columna] != categoria_excl]
    topics_entreno = pd.array(df_cat_new[columna]) 
    df_train, df_test, y_train, y_test = train_test_split(df_cat_new, topics_entreno,test_size=percent, random_state=1)
    return  df_train,df_test,y_train,y_test


# Another function to balance the number of records in different categories.
def balancing_values(df, column, num_bal):
    cat_counts = df[column].value_counts()
    cat_values = cat_counts.values
    
    num = len(cat_values)
    values = cat_values[:num]  # Take the first 'num' values
    mini = np.min(values)
    total_sum = np.sum(values)

    rel = np.max(values) / mini
    p_values = values / total_sum

    new_sum = total_sum / (rel / num_bal) + mini
    new_rel = mini / new_sum
    new_values = [p_val * new_sum if p_val > new_rel else p_val * new_sum + mini for p_val in p_values]

    return tuple(np.round(new_values))

    

# Items in each category after balancing.
def final_cat_balancing(df, num_cat, cat):   # It has worked with 5 categories, needs testing with more
    cat_counts = df[cat].value_counts()
    cat_values = cat_counts.index

    # Create a list of DataFrames containing the first 'num_val.round()' rows for each unique value in 'cat'
    dfs = [df[df[cat] == c].iloc[:int(num_val.round())] for c, num_val in zip(cat_values, num_cat)]

    # Concatenate the DataFrames
    concatenated_df = pd.concat(dfs)
    return concatenated_df


# Training with two different models, the first one Logistic Regresion and the second Stochastic Gradient Descend.
# We chose these hyperparameters because they provided the best fit for our specific use case.
def train_model_LG(df, text_column, labels, vectorizer, model):
    TF_vectorizer_train_sw_s = vectorizer(min_df=2, max_df=0.95, max_features=1200)
    TF_vectorizer_train_sw_s.fit(df[text_column])
    tf_train = TF_vectorizer_train_sw_s.transform(df[text_column])
    tf_train = tf_train.toarray()

    clf_log_train_sw_s_tf = model(max_iter=2000)
    clf_log_train_sw_s_tf.fit(tf_train, labels)
    return TF_vectorizer_train_sw_s, clf_log_train_sw_s_tf

def train_model_SGD(df, text_column, labels, vectorizer, model):
    TF_vectorizer_train_sw_s = vectorizer(min_df=2, max_df=0.95, max_features=1200)
    TF_vectorizer_train_sw_s.fit(df[text_column])
    tf_train = TF_vectorizer_train_sw_s.transform(df[text_column])
    tf_train = tf_train.toarray()

    clf_log_train_sw_s_tf = model(max_iter=2000, loss='log')
    clf_log_train_sw_s_tf.fit(tf_train, labels)
    return TF_vectorizer_train_sw_s, clf_log_train_sw_s_tf


# Viewing the obtained results.
def evaluate_model(model, vectorizer, data, labels):
    tf_data = vectorizer.transform(data)
    tf_data = tf_data.toarray()
    predictions = model.predict(tf_data)

    # Getting probabilities
    probabilities = model.predict_proba(tf_data).round(2)
    probabilities = probabilities.tolist()
    #plot_confusion_matrix(model, tf_data, labels)   PROBLEMATIC ISSUE WITH CONFUSION MATRIX
    #plt.show()
    pred_probabilities = model.predict_proba(normalize(tf_data))
    print('')
    print(classification_report(labels, predictions))
    return predictions, probabilities


# Simply predictions and probabilities.
def model_results(model, vectorizer, data):
    tf_data = vectorizer.transform(data)
    tf_data = tf_data.toarray()
    predictions = model.predict(tf_data)
    probabilities = model.predict_proba(tf_data).round(2)
    probabilities = probabilities.tolist()
    return predictions, probabilities


#This function is highly specific to our particular use case, where certain records may be 
# classified as 'GENERIC', indicating they lack sufficient information to be categorized under 
# any of our existing labels. We've provided it in Spanish as an example; 
# you can modify the RegEx pattern according to your specific use case."
def pres_gen(texto, num_pal, categoria_actual,pat_especifico,desconocido):
    if categoria_actual == desconocido and num_pal < 4:
        if re.search(r'\b(prest)\w*\b', texto) and re.search(r'\b(inform)\w*\b', texto) and re.search(r'\b(sobre)\w*\b', texto):
            return 'GENERICO' 
        elif re.search(r'\b(prest)\w*\b', texto) and re.search(r'\b(inform)\w*\b', texto) and re.search(r'\b(consult|solicit)\w*\b', texto):
            return 'GENERICO'
        else:
            return categoria_actual  
    elif categoria_actual == desconocido and num_pal >= 4:
        if re.search(pat_especifico, texto):
            return 'GENERICO' 
        else:
            return categoria_actual
    else:
        return categoria_actual 


# We train and evaluate our models using the SMOTE technique so that our disbalanced categories are balanced.
def smote_train_and_evaluate(df_train, text_column, y_train, vectorizer, model, category, num):  
    TF_vectorizer_train_sw_s = vectorizer(min_df=2, max_df=0.95, max_features=1200) 
    TF_vectorizer_train_sw_s.fit(df_train[text_column])
    tf_train_sw_s = TF_vectorizer_train_sw_s.transform(df_train[text_column])
    tf_train_sw_s = tf_train_sw_s.toarray()

    sampling_strategy_dict = {category: num}   # category needs to be provided as a string
    smote = SMOTE(sampling_strategy=sampling_strategy_dict)
    tf_train_sw_s_smote, y_train_balance_smote = smote.fit_resample(tf_train_sw_s, y_train) 

    clf_log_train_sw_s_tf = model(max_iter=2000, solver='lbfgs') # train models
    clf_log_train_sw_s_tf.fit(tf_train_sw_s_smote, y_train_balance_smote)

    tf_data = TF_vectorizer_train_sw_s.transform(data)
    tf_data = tf_data.toarray()
    predictions = clf_log_train_sw_s_tf.predict(tf_data)

    # Getting probabilities
    probabilities = clf_log_train_sw_s_tf.predict_proba(tf_data).round(2)
    probabilities = probabilities.tolist()
    
    #plot_confusion_matrix(model, tf_data, labels)   PROBLEMATIC ISSUE WITH CONFUSION MATRIX
    #plt.show()

    pred_probabilities = clf_log_train_sw_s_tf.predict_proba(normalize(tf_data))
    print('')
    print(classification_report(labels, predictions))
    return predictions, probabilities  


# This function allows us to set thresholds for the probabilities of belonging to the categories. 
# This way, when there are doubts about assigning a category, we can choose not to assign any.
def thresholds(df, threshold_cat, threshold_diff, col_max, col_2_max, Cat_2_Manual_Level_desc, unknown):
    for index, row in df.iterrows():
        if abs(row[col_max] - row[col_2_max]) < threshold_diff:
            df.at[index, Cat_2_Manual_Level_desc] = unknown
        elif abs(row[col_max]) < threshold_cat:
            df.at[index, Cat_2_Manual_Level_desc] = unknown
    return df


# Validation using F1 score:
def f1_min(df, man_cat, pred_cat):  
    unique_categories = df[man_cat].unique() 
    classification_report_dict = classification_report(df[man_cat], df[pred_cat], output_dict=True)
    f1_scores_per_category = {category: classification_report_dict[category]['f1-score'] for category in classification_report_dict if category != 'accuracy'}
    f1_final = []
    for category, f1_score in f1_scores_per_category.items():
        f1_final.append(f1_score)
    f1_final = f1_final[0:len(unique_categories)]
    return f1_final


# Re-trainind depending on the number of rows.
def reentreno(df,num):
    if df.shape[0]>num:
        return True
    else:
        return False


# A class and a function to make a validation based on the f1-score to see the performance of the model.
# If the f1-score is higher than the threshold determined, the new model will be used, and if not, the old one.
class Manual_Level_1_Cat:
    def __init__(self, df: pd.DataFrame, excluded_cat, cat_quotes):
        self.df = df 
        self.excluded_cat = excluded_cat
        self.cat_quotes = cat_quotes

    def manual_cat(self):
        self.df['Num_Words'] = self.df[self.cat_quotes].apply(lambda x: x.count(' ')+1)  
        self.df['Clean_Quotes_Dict'] = self.df[self.cat_quotes].apply(lambda x: dicc_pal(x))
        self.df["Level_1_Cat"] = self.df.Clean_Quotes_Dict.map(create_cat_1_level)   
        self.df = self.df[self.df.Level_1_Cat != self.excluded_cat]  
        return self.df  

def train_and_validate_1st_level(df, num_retrain, retrain, excluded_cat, cat_quotes, percentage, vectorizer, model, f1_score, path_old_vect, path_old_model, path_new_vect, path_new_model):
    if retrain(df, num_retrain): 
        df = Manual_Level_1_Cat(df, excluded_cat, cat_quotes).manual_cat()
        df_train, df_test, y_train, y_test = tr_te_split(df, 'Level_1_Cat', excluded_cat, percentage)

        trained_vectorizer, trained_model = train_model_SGD(df_train, 'clean_text_stopwords_stemming', y_train, vectorizer, model)
        pred, prob = evaluate_model(trained_model, trained_vectorizer, df_test['clean_text_stopwords_stemming'], y_test)
        df_test['PRED_CATEGORY'] = pred

        f1_scores = f1_min(df_test, 'Level_1_Cat', 'PRED_CATEGORY')
        if min(f1_scores) > f1_score:
            joblib.dump(trained_vectorizer, path_new_vect)
            joblib.dump(trained_model, path_new_model)
            print('Models have been updated')
            return path_new_vect, path_new_model
        else:
            print('Models have not been updated')
            return path_old_vect, path_old_model
    else: 
        print('No retraining')
        return path_old_vect, path_old_model


# A function that calculates that day when the model is executed.
def date(df, column):
    if len(sys.argv) > 1:
        provided_date = sys.argv[1]
        provided_date = datetime.strptime(provided_date, '%Y-%m-%d').date()
        print("Provided Date:", provided_date)
    else:
        provided_date = datetime.now().date() - timedelta(days=1)
        print("Taking yesterday's date")
    df[column] = pd.to_datetime(df[column])
    df[column] = df[column].dt.date
    rows_with_provided_date = df[df[column] == provided_date]    
    if rows_with_provided_date.empty:
        print("The DataFrame is empty. Exiting the script.")
        sys.exit()  
    return rows_with_provided_date 


   