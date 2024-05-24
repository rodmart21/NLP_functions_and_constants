import re
from src.utils import * # cargamos las constantes
from nltk.stem import SnowballStemmer
import spacy
from nltk.corpus import stopwords
import sys
import pandas as pd

# Cleaning special characters in catalan.
pat_cat=r"[àèò'ç·]"
pat_cat=re.compile(pat_cat, re.I)

# Substitution of any character not used in spanish.
pat_keep=r"[^a-zñáéíóú\s]"
pat_keep=re.compile(pat_keep, re.I)

# Substitution of accents.
dict_sub_accent = {"á": "a", "é":"e", "í":"i", "ó":"o", "ú": "u"}

# Some letters are allowed to be repeated in spanish.
letters_allow_rep= {"c", "e", "l", "o", "r", "n"}
pat= r"(\w)\1"    
pat_cons_rep= r"[^ aeiou]{4,}"

# Pattern for eliminating repeated words. 
pat_pal= r"(\b\w+( \w+)*?\b)(\s\1)+"    

# Elimination of repeated vowels.
pat_vow = r"(^[aeiou]{3,})"

# Lematize and stemming for Spanish.
STEMMER_ES = SnowballStemmer("spanish")
SPACY_NLP_ES = spacy.load("es_core_news_sm")

# Elimination of accents inside the stopwords.
STOPWORDS = stopwords.words('spanish')
stop_words_notilde= [remove_accents(word,dict_sub_accent) for word in STOPWORDS]

# Patterns used specific of this use case to identify the generic categories.
pat_especifico_prest =  r'\b(solicitud de otras prestaciones|consulta sobre prestaciones|informacion sobre una prestacion|solicitude otras prestaciones|informacion de prestacion|informacion sobre mi prestacion |prestaciones y otras gestiones|prestaciones de la seguridad social|prestaciones seguridad social)\w*\b'  
pat_especifico_pens =  r'\b(informacion sobre mi pension|informacion sobre la pension|consulta sobre mi pension|pensiones y otras prestaciones|consulta sobre la pension|informacion sobre las pensiones|informacion de la pension|informacion de mi pension|solicitar certificado de pensiones|solicitar informacion sobre pensiones|informacion solicitud pension|solicitud de pensiones y otras prestaciones|solicitar informacion sobre la pension)\w*\b'