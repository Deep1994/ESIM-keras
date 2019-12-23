# -*- coding: utf-8 -*-
"""
preprocess the dataset
"""
import re

def clean_text(text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """

        # SPECIAL_TOKENS = {
        #     'quoted': 'quoted_item',
        #     'non-ascii': 'non_ascii_word',
        #     'undefined': 'something'
        # }

        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)
        # remove comma between numbers, i.e. 15,000 -> 15000
        text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        # text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)
        text = re.sub(r"\%", " percent ", text)   

        # replace the float numbers with a random number
        # text = re.sub(r"\d+\.\d+", "666", text)

        # replace non-ascii word with special word
        # def pad_str(s):
        #     return ' '+s+' '
        # text = re.sub(r"[^\x00-\x7F]+", pad_str(SPECIAL_TOKENS['non-ascii']), text)

        # indian dollar
        text = re.sub(r"(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
        text = re.sub(r" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
        # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
        text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
        text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
        text = re.sub(r" india ", " India ", text)
        text = re.sub(r" switzerland ", " Switzerland ", text)
        text = re.sub(r" china ", " China ", text)
        text = re.sub(r" chinese ", " Chinese ", text) 
        text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
        text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
        text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
        text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
        text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
        text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
        text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
        text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
        text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
        text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
        text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
        text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
        text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
        text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
        text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
        text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
        text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
        text = re.sub(r" III ", " 3 ", text)
        text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
        text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
        text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
        
        # remove extra space
        text = ' '.join(text.split())

        return text


    
    
    
    



















