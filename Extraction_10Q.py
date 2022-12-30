import re
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import pandas as pd

vowels = "aeiouy"

stop_words = ['ME', 'MY', 'MYSELF', 'WE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'OUR', 'OURS', 'OURSELVES', 'YOU', 'YOUR', 'YOURS','AND','THE'
          'YOURSELF', 'YOURSELVES', 'HE', 'HIM', 'HIS', 'HIMSELF', 'SHE', 'HER', 'HERS', 'HERSELF',
          'IT', 'ITS', 'ITSELF', 'THEY', 'THEM', 'THEIR', 'THEIRS', 'THEMSELVES', 'WHAT', 'WHICH',
          'WHO', 'WHOM', 'THIS', 'THAT', 'THESE', 'THOSE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'BE',
          'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'HAVING', 'DO', 'DOES', 'DID', 'DOING', 'AN',
          'BUT', 'IF', 'OR', 'BECAUSE', 'AS', 'UNTIL', 'WHILE', 'OF', 'AT', 'BY',
          'FOR', 'WITH', 'ABOUT', 'BETWEEN', 'INTO', 'THROUGH', 'DURING', 'BEFORE',
          'AFTER', 'ABOVE', 'BELOW', 'TO', 'FROM', 'UP', 'DOWN', 'IN', 'OUT', 'ON', 'OFF', 'OVER',
          'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'THERE', 'WHEN', 'WHERE', 'WHY',
          'HOW', 'ALL', 'ANY', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME', 'SUCH',
          'NO', 'NOR', 'NOT', 'ONLY', 'OWN', 'SAME', 'SO', 'THAN', 'TOO', 'VERY', 'CAN',
          'JUST', 'SHOULD', 'NOW', 'AMONG','NBSP']




mda_start1 = r"(\n*^\s*[item\s\n]+\s*\n*[2]\s*\n*[-‒–—.:\s]+\s*\n*(KMART\s*\n*\s*CORPORATION)?\s*\n*\s*((MANAGEMENT\s*\n*\s*.\s*\n*\s*S)|(MANAGEMENTS.))\s*\n*(DISCUSSION)?\s*\n*(AND)?\s*\n*(ANALYSIS)?\s*\n*)"
mda_end1 = r"(\n*^\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*\s*)" \
           r"|(\n*^\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Qualitative\s*\n*and\s*\n*Quantitative\s*\n*\s*)" \
           r"|(\n*^\s*[item\s\n]+\s*\n*\s*[4]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Controls\s*\n*\s*and\s*\n*\s*Procedures\s*\n*\s*)" \
           r"|(\n*^\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*OTHER\s*\n*\s*INFORMATION\s*\n*\s*[:]?s*\n*\s*)" \
           r"|(\n*^\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])$)"  \
           r"|(\n*^\s*[item\s\n]+\s*\n*\s*[1]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*LEGAL\s*\n*\s*PROCEEDING(S)?\s*\n*\s*)" \
           r"|(\n*^\s*[item\s\n]+\s*\n*\s*[6]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*[Exhibits\s]+\s*\n*\s*)" \
            r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
           r"|^EOF$" 
        


mda_start2 = r"(\n*\s*[item\s\n]+\s*\n*[2]\s*\n*[-‒–—.:\s]+\s*\n*(KMART\s*\n*\s*CORPORATION)?\s*\n*\s*((MANAGEMENT\s*\n*\s*.\s*\n*S)|(MANAGEMENTS.))\s*\n*DISCUSSION\s*\n*AND\s*\n*ANALYSIS\s*\n*)"
mda_end2 = r"(\n*\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*\s*)" \
           r"|(\n*\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Qualitative\s*\n*and\s*\n*Quantitative\s*\n*\s*)" \
           r"|(\n*\s*[item\s\n]+\s*\n*\s*[4]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Controls\s*\n*\s*and\s*\n*\s*Procedures\s*\n*\s*)" \
           r"|(\n*\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*OTHER\s*\n*\s*INFORMATION\s*\n*\s*[:]?s*\n*\s*)" \
           r"|(\n*\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])$)"  \
           r"|(\n*\s*[item\s\n]+\s*\n*\s*[1]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*LEGAL\s*\n*\s*PROCEEDING(S)?\s*\n*\s*)" \
           r"|(\n*\s*[item\s\n]+\s*\n*\s*[6]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*[Exhibits\s]+\s*\n*\s*)" \
            r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
           r"|^EOF$" 

          
mda_start3 = r"(\n*\s*([2]\s*\n*\s*[-‒–—.:\s]+\s*\n*)?\s*\n*\s*MANAGEMENT\s*\n*\s*.\s*\n*\s*(S)?\s*\n*\s*(FINANCIAL)?\s*\n*\s*DISCUSSION\s*\n*\s*(AND)?\s*\n*\s*ANALYSIS\s*\n*(of)?\s*\n*\s*(the)?\s*\n*\s*(FINANCIAL)?\s*\n*\s*(CONDITION)?\s*\n*\s*(AND)?\s*\n*\s*(RESULTS)?\s*\n*\s*(OF)?\s*\n*\s*(OPERATIONS)?$)" \
             r"|(\n*\s*([2]\s*\n*\s*[-‒–—.:\s]+\s*\n*)?\s*\n*\s*MANAGEMENT\s*\n*\s*.\s*\n*(S)?\s*\n*\s*\n*\s*ANALYSIS\s*\n*(of)?\s*\n*\s*(the)?\s*\n*\s*(FINANCIAL)?\s*\n*\s*(CONDITION)?\s*\n*\s*(AND)?\s*\n*\s*(RESULTS)?\s*\n*\s*(OF)?\s*\n*\s*(OPERATIONS)?$)" \
            r"|(MANAGEMENT'S\s*\n*\s*DISCUSSION\s*\n*\s*AND\s*\n*\s*ANALYSIS\s*\n*\s*OF\s*\n*\s*RESULTS\s*\n*\s*OF\s*\n*\s*OPERATIONS\s*\n*\s*AND\s*\n*\s*FINANCIAL\s*\n*\s*CONDITION)"

mda_end3 = r"(\n*\s*([item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+)?\s*\n*\s*QUANTITATIVE\s*\n*\s*AND\s*\n*\s*QUALITATIVE\s*\n*\s*DISCLOSURE(S)?\s*\n*\s*ABOUT\s*\n*\s*MARKET\s*\n*\s*RISK\s*\n*\s*$)" \
           r"|(\n*^\s*([item\s\n]+\s*\n*[1]\s*\n*[-‒–—.:\s]+)?LEGAL\s*\n*\s*PROCEEDINGS\s*\n*\s*$)" \
           r"|(\n*\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*(OTHER\s*\n*\s*INFORMATION)?\s*\n*\s*[:]?s*\n*\s*)"\
           r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
           r"|(\n*^\s*EXECUTION\s*\n*\s*COPY\s*\n*\s*$)"\
           r"|(\n*\s*-----END PRIVACY-ENHANCED MESSAGE-----\s*\n*\s*)" \
           r"|^EOF$"
           







rf_start1 = r"(\n*^\s*[item\s\n]+\s*\n*[1]\s*\n*\(?[A]\)?\s*\n*\s*[-‒–—.:\s]+\s*\n*R\s*\n*\s*I\s*\n*\s*S\s*\n*\s*K\s*\n*\s*)"
rf_end1 = r"(\n*^\s*[item\s\n]+\s*\n*[2]\s*\n*\(?[C]?\)?\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*((Unregistered\s*\n*\s*Sales\s*\n*\s*of\s*\n*\s*Equity)|(Purchases\s*\n*\s*of\s*\n*\s*Equity\s*\n*\s*Securities))\s*\n*\s*)" \
          r"|(\n*^\s*[item\s\n]+\s*\n*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Defaults\s*\n*\s*Upon\s*\n*\s*Senior\s*\n*\s*)" \
          r"|(\n*^\s*[item\s\n]+\s*\n*[4]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*((Mine\s*\n*\s*Safety\s*\n*\s*)|(SUBMISSION\s*\n*\s*OF\s*\n*\s*MATTERS\s*\n*\s*)))" \
          r"|(\n*^\s*[item\s\n]+\s*\n*[5]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Other\s*\n*\s*Information\s*\n*\s*)" \
          r"|(\n*^\s*[item\s\n]+\s*\n*[6]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*[Exhibits\s]+\s*\n*\s*)" \
            r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
            r"|^EOF$"


rf_start2 = r"(\n*\s*[item\s\n]+\s*\n*[1]\s*\n*\(?[A]\)?\s*\n*\s*[-‒–—.:\s]+\s*\n*R\s*\n*\s*I\s*\n*\s*S\s*\n*\s*K\s*\n*\s*)"
rf_end2 = r"|(\n*\s*[item\s\n]+\s*\n*[2]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Unregistered\s*\n*\s*Sales\s*\n*\s*of\s*\n*\s*Equity\s*\n*\s*)" \
          r"|(\n*\s*[item\s\n]+\s*\n*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Defaults\s*\n*\s*Upon\s*\n*\s*Senior\s*\n*\s*)" \
          r"|(\n*\s*[item\s\n]+\s*\n*[4]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Mine\s*\n*\s*Safety\s*\n*\s*)" \
          r"|(\n*\s*[item\s\n]+\s*\n*[5]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Other\s*\n*\s*Information\s*\n*\s*)" \
          r"|(\n*\s*[item\s\n]+\s*\n*[6]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*[Exhibits\s]+\s*\n*\s*)" \
            r"|(\n*\s*Signature(s)\s*\n*\s*$)" \
          r"|^EOF$"


rf_start3 = r"(\n*^\s*RISK\s*\n*\s*Factors\s*\n*\s*$3)"
rf_end3 = r"(\n*\s*Unregistered\s*\n*\s*Sales\s*\n*\s*of\s*\n*\s*Equity\s*\n*\s*$)" \
          r"|(\n*\s*Defaults\s*\n*\s*Upon\s*\n*\s*Senior\s*\n*\s*$)" \
          r"|(\n*\s*Mine\s*\n*\s*Safety\s*\n*\s*$)" \
          r"|(\n*^\s*Other\s*\n*\s*Information\s*\n*\s*$)" \
          r"|(\n*\s*[Exhibits\s]+\s*\n*\s*$)" \
            r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
          r"|^EOF$"



qqdmr_start1 = r"(\n*^\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*)" \
             r"|(\n*^\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Qualitative\s*\n*\s*and\s*\n*\s*Quantitative)"
qqdmr_end1 =r"(\n*^\s*[item\s\n]+\s*\n*\s*[4]\s*\n*\s*[T]?\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Controls\s*\n*\s*and\s*\n*\s*Procedures\s*\n*\s*)" \
            r"|(\n*^\s*INDEPENDENT\s*\n*\s*ACCOUNTANTS.\s*\n*\s*REPORT\s*\n*\s*)" \
            r"|(\n*^\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*OTHER\s*\n*\s*INFORMATION\s*\n*\s*[:]?s*\n*\s*)" \
            r"|(\n*^\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])$)" \
                r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
                r"|^EOF$"

qqdmr_start2 =  r"(\n*\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*)" \
                r"|(\n*\s*[item\s\n]+\s*\n*\s*[3]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Qualitative\s*\n*\s*and\s*\n*\s*Quantitative)"

qqdmr_end2 = r"(\n*\s*[item\s\n]+\s*\n*\s*[4]\s*\n*\s*[T]?\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Controls\s*\n*\s*and\s*\n*\s*Procedures\s*\n*\s*)" \
             r"|(\n*\s*INDEPENDENT\s*\n*\s*ACCOUNTANTS.\s*\n*\s*REPORT\s*\n*\s*)" \
             r"|(\n*\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*OTHER\s*\n*\s*INFORMATION\s*\n*\s*[:]?s*\n*\s*)" \
             r"|(\n*\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2]))" \
                r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
                r"|^EOF$"

qqdmr_start3 = r"(\n*\s*QUANTITATIVE\s*\n*\s*AND\s*\n*\s*QUALITATIVE\s*\n*\s*DISCLOSURES\s*\n*\s*ABOUT\s*\n*\s*MARKET\s*\n*\s*RISK\s*\n*\s*$)" \
    r"|(\n*\s*QUALITATIVE\s*\n*\s*AND\s*\n*\s*QUANTITATIVE\s*\n*\s*DISCLOSURES\s*\n*\s*ABOUT\s*\n*\s*MARKET\s*\n*\s*RISK\s*\n*\s*$)"

qqdmr_end3 = r"(\n*\s*PART\s*\n*\s*(([I]\s*\n*\s*[I])|[2])\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*(OTHER\s*\n*\s*INFORMATION)?\s*\n*\s*[:]?s*\n*\s*)" \
             r"(\n*\s*([item\s\n]+\s*\n*\s*[4]\s*\n*\s*[T]?\s*\n*\s*[-‒–—.:\s]+)?\s*\n*\s*Controls\s*\n*\s*and\s*\n*\s*Procedures\s*\n*\s*$)" \
             r"|(\n*\s*-----END PRIVACY-ENHANCED MESSAGE-----\s*\n*\s*$)" \
                r"|(\n*^\s*Signature(s)\s*\n*\s*$)" \
                r"|^EOF$"




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
        pass

    if word.endswith("ed"):
        count -= 1
    if count == 0:
        count += 1
    return count



def mda_extract(text):
    output = []
    try:
        #print("1")

        for match in re.finditer(mda_start1, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            start = match.start()
            #print(match.group())
            mend = re.search(mda_end1, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            #print(mend.group())
            output.append(text[start:start + mend.start()])

        #print("2")
        if len(output)<1:
            for match in re.finditer(mda_start2, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                #print(match.group())
                mend = re.search(mda_end2, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend.group())
                output.append(text[start:start + mend.start()])
        #print("3")   
        if len(output)<1 or len(max(output, key=len))<200:
            for match in re.finditer(mda_start3, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                #print(match.group())
                mend = re.search(mda_end3, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend.group())
                output.append(text[start:start + mend.start()])

        words = word_tokenize(max(output, key=len) )
        #print(len(words))
        words = [word.lower() for word in words if word.isalpha()]
        words = [word.lower() for word in words if not word.upper() in stop_words]
        
        ##print(max(output, key=len))

        wc = len(words)
        cwc = 0
        for word in words:
            if syllable_count(word) > 2:
                cwc += 1
        return([wc,cwc])
    except Exception as e:
        #print(e)
        return ([0,0])





def rf_extract(text):

    try:
        output = []
        #print("1")

        for match in re.finditer(rf_start1, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            start = match.start()
            #print(match.group())
            mend = re.search(rf_end1, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            #print(mend.group())
            output.append(text[start:start + mend.start()])
        #print("2")
        if len(output)<1:
            for match in re.finditer(rf_start2, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                #print(match)
                mend = re.search(rf_end2, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend.group())
                output.append(text[start:start + mend.start()])
        #print("3")
        if len(output)<1 :
            for match in re.finditer(rf_start3, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                #print(match)
                mend = re.search(rf_end3, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend.group())
                output.append(text[start:start + mend.start()])
        ##print(max(output, key=len))
        words = word_tokenize(max(output, key=len))
        words = [word.lower() for word in words if word.isalpha()]
        words = [word.lower() for word in words if not word.upper() in stop_words]

        wc = len(words)
        cwc = 0
        for word in words:
            if syllable_count(word) > 2:
                cwc += 1
        return (wc,cwc)
    except:
        return (0,0)





def qqdmr_extract(text):

    try:
        output = []
        #print("1")

        for match in re.finditer(qqdmr_start1, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):         
            #print(match)
            start = match.start()
            mend = re.search(qqdmr_end1, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            #print(mend)
            output.append(text[start:start + mend.start()])

        #print("2")
        if len(output)<1:
            for match in re.finditer(qqdmr_start2, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                #print(match)
                start = match.start()
                mend = re.search(qqdmr_end2, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend)
                output.append(text[start:start + mend.start()])

        #print("3")
        if len(output)<1 or len(max(output, key=len))<200:
            for match in re.finditer(qqdmr_start3, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                #print(match)
                start = match.start()
                mend = re.search(qqdmr_end3, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend.group())
                output.append(text[start:start + mend.start()])
                
        #print(max(output, key=len))
        words = word_tokenize(max(output, key=len))
        words = [word.lower() for word in words if word.isalpha()]
        words = [word.lower() for word in words if not word.upper() in stop_words]

        wc = len(words)
        cwc = 0
        for word in words:
            if syllable_count(word) > 2:
                cwc += 1
        return (wc,cwc)
    except:
        return (0,0)


