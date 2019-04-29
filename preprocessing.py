import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

df = pd.read_csv("cleaned_db.csv")

# fatal_non = {"Fatal":1,
#             "Nonfatal":0}
# df["Degree of Injury"]=df["Degree of Injury"].map(fatal_non)
# df["Abstract Text"]=df["Abstract Text"].apply(lambda x: re.sub("[^a-zA-Z]"," ",str(x)))

df['abstract_text'] = df.apply(lambda row: word_tokenize(row['abstract_text']), axis=1)
df["abstract_text"] = df['abstract_text'].apply(lambda x: [item for item in x if item.isalpha()])

df['abstract_text'] = df['abstract_text'].apply(lambda x : [WordNetLemmatizer().lemmatize(y) for y in x])

stop = stopwords.words('english')
new_stop = ["employee", "die", "injury", "injured","ft","feet", "work","killed","wa", "january","february","march","april","may","june","july",
            "august","september","october","november","december","morning","afternoon","dead", "hospital","approximately","hospitalized","medical","center"
            "service", "mobile", "narrative", "information", "died", "detail", "performing", "regular", "hospitalization", "serial","worker","site","concrete",
            "called", "emergency","foot", "number","balance", "osha","jurisdiction", "fracture","toilet", "deceased", "reported", "cookie","original","noted",
            "describe", "blunt", "initial", "related", "suffered", "sustained","however", "left","right","index", "middle","incident","transported","center"
            "treated","pinky","partial","tip","treatment","became","ring","got","service","admitted","surgical","procedure","height","company","construction", "determined"
            , "specified"]
stop.extend(new_stop)
not_stopwords = {'not', 'above', 'below','against','between','miss'}
final_stop_words = set([word for word in stop if word not in not_stopwords])
df['abstract_text'] = df['abstract_text'].apply(lambda x: [item for item in x if item not in final_stop_words])

df.to_csv("processed_stop.csv", index=True)
print("c")