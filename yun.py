import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize
import Extraction_10k as exrk, Extraction_10Q as exrq,other_forms as ofrm



def syllable_count(word):
    word = word.lower()

    count = 0

    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("es"):
        count -= 1
    if word.endswith("e"):
        # print(word)
        count -= 1
    if word.endswith("ed"):
        count -= 1
    if count == 0:
        count += 1
    return count


vowels = "aeiouy"

stop_words = ['me', 'my', 'myself', 'we', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
              'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
              'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'and', 'the',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
              'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
              'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'an',
              'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
              'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
              'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
              'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
              'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
              'just', 'should', 'now', 'among', 'nbsp']



def countall(content):
    try:
        text = content.prettify()
    except:
        text = " ".join(content.findAll(text=True))

    text = re.sub(r'&lt;', '<', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    text = re.sub(r'&gt;', '>', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    pattern = r'\<\s*\n*\s*TYPE\s*\n*\s*\>\s*\n*\s*(GRAPHIC|ZIP|EXCEL|JSON|PDF|xml)\s*\n*\s*[\W\w\n ]*?\<\s*\n*\s*\/\s*\n*\s*document\s*\n*\s*\>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE <SCRIPT> to </script> and variations)
    pattern = r'<[ ]*script[\W\w\n ]*?\/[ ]*script[ ]*>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text.lower(), flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE <SCRIPT> to </script> and variations)
    pattern = r'<[ ]*xml[\W\w\n ]*?\/[ ]*xml[ ]*>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text.lower(), flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML <STYLE> to </style> and variations)
    pattern = r'<[ ]*style[\W\w\n ]*?\/[ ]*style[ ]*>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML <META> to </meta> and variations)
    pattern = r'<[ ]*meta[\W\w\n ]*?>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML COMMENTS <!-- to --> and variations)
    pattern = r'<[ ]*!--[\W\w\n ]*?--[ ]*?>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML DOCTYPE <!DOCTYPE html to > and variations)
    pattern = r'<[ ]*\![ ]*DOCTYPE[\W\w\n ]*?>'  # mach any char zero or more times
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    pattern = r'<[ ]*us-gaap:[\W\w\n ]*?\/[ ]*us-gaap[ ]*[\W\w\n ]*?>'  # mach any char zero or more times<XBRL>
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    pattern = r'<[ ]*XBRL[\W\w\n ]*?\/[ ]*XBRL[ ]*>'  # mach any char zero or more times<XBRL>
    text = re.sub(pattern, ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    text = re.sub(r"<[\W\w\n ]*?>", ' ', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))


    text = re.sub(r'&#\d+;', ' ', text)
    text = text.replace(" ", " ")
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\n+\s+', '\n', text)
    text = text.lower()

    words = word_tokenize(text)
    words = [word for word in words if word.isalpha() and len(word) > 3]
    words = [word for word in words if not word in stop_words]

    wc = len(words)
    cwc = 0
    for word in words:
        if syllable_count(word) > 2:
            cwc += 1

    return wc,cwc


if __name__ == '__main__':

    sheet = pd.read_excel('input2.xlsx')
    for index, rows in sheet.iterrows():
        if index%10==0:
            sheet.to_excel("output-all.xlsx")

        try:
            url = "https://www.sec.gov/Archives/" + sheet.iloc[index, 5]
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}
            request = requests.get(url, headers=headers)
            html = request.content
            content = BeautifulSoup(html, 'html.parser')
            wc,cwc=countall(content)

            html_regex = re.compile(r'<.*>')
            try: 
                text=content.find('text').prettify()
                text = re.sub(html_regex, ' ',text)
            except:
                try:    
                    text=content.find('body').prettify()
                    text = re.sub(html_regex, ' ', text)
                except:
                    text = content.findAll(text=True)
                    text = re.sub( html_regex, ' ', " ".join(text))

            text = re.sub(r'&#\d+;', ' ', text)
            text = re.sub(r' +', ' ', text)
            text = text.replace(" ", " ")
            text = re.sub(r'\n+\s+', '\n', text)


            form_type = str(sheet.iloc[index, 4])
            if form_type.upper().startswith("10-Q") or form_type.upper().startswith("10Q"):
                mda_wc,mda_cwc  =  exrq.mda_extract(text)
                qqdmr_wc,qcwc  =  exrq.qqdmr_extract(text)
                rf_wc,rf_cwc  = exrq.rf_extract(text)


            elif form_type.upper().startswith("10-K") or form_type.upper().startswith("10K"):
                mda_wc,mda_cwc  =  exrk.mda_extract(text)
                qqdmr_wc,qcwc  =  exrk.qqdmr_extract(text)
                rf_wc,rf_cwc  = exrk.rf_extract(text)

            #inprogress
            else:
                mda_wc,mda_cwc  =  ofrm.mda_extract(text)
                qqdmr_wc,qcwc  =  ofrm.qqdmr_extract(text)
                rf_wc,rf_cwc  = ofrm.rf_extract(text)
            #inprogress
                  
            data = [ qqdmr_wc,
                    qcwc,
                    rf_wc,
                    rf_cwc,
                    mda_wc,
                    mda_cwc,
                    wc,
                    cwc
                ]


            print(url,data)


            sheet.iloc[index,10:] = data
            print(index)
            print("\n\n=====================================================================================================\n\n")
        except Exception as e:
            data = [-1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1
                ]
            sheet.iloc[index,10: ] = data
            print(e)


    sheet.to_excel("output-all.xlsx")