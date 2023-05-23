# import pandas as pd
# import numpy as np
# import tensorflow.keras
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model

# train_data = pd.read_csv("./static/assets/dataset/updated_product_dataset.csv")
# training_sentences = []

# for i in range(len(train_data)):
#     sentence = train_data.loc[i, "Text"]
#     training_sentences.append(sentence)

# model = load_model("./static/assets/model/Customer_Review_Text_Emotion.h5")

# vocab_size = 40000
# max_length = 100
# trunc_type = "post"
# padding_type = "post"
# oov_tok = "<OOV>"

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(training_sentences)

# # word_index = tokenizer.word_index

# # training_sequences = tokenizer.texts_to_sequences(training_sentences)
# # training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# # dictionary where key : emotion , value : list
# encode_emotions = {
#                     "Neutral": [0,"./static/assets/emoticons/neutral.png"],
#                     "Positive": [1,"./static/assets/emoticons/positive.png"],
#                     "Negative": [2,"./static/assets/emoticons/negative.png"]
#                     }


# def predict(text):

#     sentiment = ""
#     emoji_url = ""
#     customer_review = []
#     customer_review.append(text)
#     sequences = tokenizer.texts_to_sequences(customer_review)
#     padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#     result = model.predict(padded)
#     label = np.argmax(result , axis=1)
#     label = int(label)

#     # extracting emotion and url from dictionary
#     for emotion in encode_emotions:
#         if encode_emotions[emotion][0]  ==  label:
#             sentiment = emotion
#             emoji_url = encode_emotions[emotion][1]

#     return sentiment


# print(predict('this is a great phone to buy'))













































# import pandas as pd
# import numpy as np

# import tensorflow.keras
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model


# train_data = pd.read_csv("./static/assets/dataset/updated_product_dataset.csv")    
# print(train_data.head())
# training_sentences = []

# for i in range(len(train_data)):
#     sentence = train_data.loc[i, "Text"]
#     training_sentences.append(sentence)

# model = load_model("./static/assets/model/Customer_Review_Text_Emotion.h5")

# vocab_size = 40000
# max_length = 100
# trunc_type = "post"
# padding_type = "post"
# oov_tok = "<OOV>"

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(training_sentences)

# emo_code_url = {
#     "Neutral": [0, "./static/assets/emoticons/neutral.png"],
#     "Positive": [1,"./static/assets/emoticons/positive.png" ],
#     "Negative": [2, "./static/assets/emoticons/negative.png"]
#     }


# def predict_review():

#     sentence = ["Great phone do buy it. It is an awesome purchase with great battery life"]
#     sequences = tokenizer.texts_to_sequences(sentence)
#     print(sequences)
#     padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#     print(padded)
#     result = model.predict(padded)
#     print(result)
#     label = np.argmax(result , axis=1)
#     label = int(label)

# #     predicted_emotion=""
# #     predicted_emotion_img_url=""
    
# #     if  text!="":
# #         sentence = []
# #         sentence.append(text)
# #         print(sentence)
# #         sequences = tokenizer.texts_to_sequences(sentence)
# #         print(sequences)

# #         padded = pad_sequences(
# #             sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
# #         )
# #         # testing_padded = np.array(padded)
# #         # print(testing_padded , testing_padded.shape)

# #         result = model.predict(padded)
# #         print(result)

# #         predicted_class_label = np.argmax(result, axis=1)        
            
# #         for key, value in emo_code_url.items():
# #             if value[0]==predicted_class_label:
# #                 predicted_emotion_img_url=value[1]
# #                 predicted_emotion=key
# #         return predicted_emotion, predicted_emotion_img_url

# # print(predict_review('Great phone do buy it. It is an awesome purchase with great battery life'))
# predict_review()

















































































































import pandas as pd
import numpy as np

import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


train_data = pd.read_csv("./static/assets/dataset/updated_product_dataset.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model/Customer_Review_Text_Emotion2.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

emo_code_url = {
    "Neutral": [0, "./static/assets/emoticons/neutral.png"],
    "Positive": [1,"./static/assets/emoticons/positive.png" ],
    "Negative": [2, "./static/assets/emoticons/negative.png"]
    }

def predict(text):

    predicted_emotion=""
    predicted_emotion_img_url=""
    
    if  text!="":
        sentence = []
        sentence.append(text)

        sequences = tokenizer.texts_to_sequences(sentence)

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=-1)        
            
        for key, value in emo_code_url.items():
            if value[0]==predicted_class_label:
                predicted_emotion_img_url=value[1]
                predicted_emotion=key
        return predicted_emotion, predicted_emotion_img_url

print(predict('Great phone, do buy it. It has long lasting battery and beautiful display'))


