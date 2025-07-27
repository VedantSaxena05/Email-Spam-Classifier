import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text     import CountVectorizer # to convert text to numerical data.
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("C:\\Users\\Lanovo\\OneDrive\\Desktop\\Spam_Class\\spam.csv")

data.drop_duplicates(inplace=True)
data['Category']=data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

mess=data['Message']
cat=data['Category']
(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess,cat,test_size=0.2)
#0.2 means 20% as test dataset and 80% as train dataset.

cv=CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#Creating the model
model=MultinomialNB()
model.fit(features,cat_train)

#Test our model
features_test = cv.transform(mess_test)
print(model.score(features_test,cat_test))

#Prediction
def predict(message):
    input_message=cv.transform([message]).toarray()
    result=model.predict(input_message)
    return result

st.header('Spam Detection')
input_mess=st.text_input('Enter message here')
if st.button('Validate'):
    output = predict(input_mess)
    if output[0] == 'Spam':
        st.error('⚠️ This message is likely **Spam**.')
    else:
        st.success('✅ This message is **Not Spam**.')










