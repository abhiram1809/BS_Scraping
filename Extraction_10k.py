import re
from nltk.tokenize import word_tokenize

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




mda_start1 = r"(\n*^\s*item\s*\n*[7]\s*\n*[-‒–—.:\s]+\s*\n*MANAGEMENT.\s*\n*S\s*\n*DISCUSSION\s*\n*AND\s*\n*ANALYSIS\s*\n*)"
mda_end1 = r"(\n*^\s*item\s*\n*\s*[7]\s*\n*\s*\(?[Aa]\)?\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*)" \
           r"|(\n*^\s*item\s*\n*\s*[8]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Financial\s*\n*\s*Statements\s*\n*\s*)" \
           r"|(\n*^\s*item\s*\n*\s*[9]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Changes\s*\n*\s*in\s*\n*\s*and\s*\n*\s*Disagreements\s*\n*\s*)" \
           r"|(\n*^\s*item\s*\n*[9]\s*\n*\(?[A-B]\)?\s*\n*[-‒–—.:\s]+\s*\n*((Controls\s*\n*\s*and\s*\n*\s*Procedures)|(Other\s*\n*\s*Information)\s*\n*\s*))" \
           r"|(\n*^\s*item\s*\n*\s*[10]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Directors.?\s*\n*\s*Executive\s*\n*\s*Officers\s*\n*\s*)" \
           r"|(\n*^\s*item\s*\n*\s*[11]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Executive\s*\n*\s*Compensation\s*\n*\s*)" \
           r"|(\n*^\s*item\s*\n*\s*[12]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Security\s*\n*\s*Ownership\s*\n*\s*of\s*\n*\s*Certain\s*\n*\s*)" \
           r"|(\n*^\s*item\s*\n*\s*[13]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Certain\s*\n*\s*Relationships\s*\n*\s*and\s*\n*\s*Related\s*\n*\s*)" \
           r"|(\n*^\s*item\s*\n*\s*[14]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Principal\s*\n*\s*Accountant\s*\n*\s*Fees\s*\n*\s*)"


mda_start2 = r"(\n*\s*item\s*\n*[7]\s*\n*[-‒–—.:\s]+\s*\n*\s*MANAGEMENT.\s*S\s*\n*)"
#mda_end2=r"(\n*\s*item\s*\n*([8-9]|1[0-4])\s*\n*\(?[A-B]?\)?\s*\n*[-‒–—.:\s]+\s*\n*)|(\n*\s*item\s*\n*[7]\s*\n*\s*.?\s*\n*\s*\(?[A]\)\s*\n*[-‒–—.:\s]+\s*\n*)"
mda_end2 = r"(\n*\s*item\s*\n*[7]\s*\n*\(?[Aa]\)?\s*\n*\s*.?\s*\n*\s*[-‒–—.:\s]+\s*\n*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*)" \
           r"|(\n*\s*item\s*\n*\s*[8]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Financial\s*\n*\s*Statements\s*\n*\s*)" \
           r"|(\n*\s*item\s*\n*\s*[9]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Changes\s*\n*\s*in\s*\n*\s*and\s*\n*\s*Disagreements\s*\n*\s*)" \
           r"|(\n*\s*item\s*\n*[9]\s*\n*\(?[A-B]\)?\s*\n*[-‒–—.:\s]+\s*\n*((Controls\s*\n*\s*and\s*\n*\s*Procedures)|(Other\s*\n*\s*Information)\s*\n*\s*))" \
           r"|(\n*\s*item\s*\n*\s*[10]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Directors.?\s*\n*\s*Executive\s*\n*\s*Officers\s*\n*\s*)" \
           r"|(\n*\s*item\s*\n*\s*[11]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Executive\s*\n*\s*Compensation\s*\n*\s*)" \
           r"|(\n*\s*item\s*\n*\s*[12]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Security\s*\n*\s*Ownership\s*\n*\s*of\s*\n*\s*Certain\s*\n*\s*)" \
           r"|(\n*\s*item\s*\n*\s*[13]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Certain\s*\n*\s*Relationships\s*\n*\s*and\s*\n*\s*Related\s*\n*\s*)" \
           r"|(\n*\s*item\s*\n*\s*[14]\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Principal\s*\n*\s*Accountant\s*\n*\s*Fees\s*\n*\s*)"


mda_start3 = r"(\n*^\s*MANAGEMENT.\s*\n*(S)?\s*\n*DISCUSSION\s*\n*(AND)?\s*\n*(ANALYSIS)?\s*\n*(of)?)"
mda_end3 = r"(\n*^\sQUANTITATIVE\s*\n*\s*AND\s*\n*\s*QUALITATIVE\s*\n*\s*DISCLOSURES\s*\n*\s*ABOUT\s*\n*\s*MARKET\s*\n*\s*RISK\s*\n*\s*$)" \
           r"(\n*^\s*item\s*\n*\s*[7]\s*\n*\s*\(?[Aa]\)?\s*\n*\s*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*and\s*\n*Qualitative\s*\n*)" \
           r"|(\n*^\s*Consolidated\s*\n*\s*Financial\s*\n*\s*of\s*\n*\s*Statements\s*\n*\s*$)" \
           r"|(\n*^\sFinancial\s*\n*\s*Statements\s*\n*\s*and\s*\n*\s*Supplementary\s*\n*\s*Data\s*\n*\s*$)" \
           r"|(\n*^\s*CONSOLIDATED\s*\n*\s*BALANCE\s*\n*\s*SHEET\s*\n*\s*$)" \
           r"|(\n*^\s*AUDITED\s*\n*\s*FINANCIAL\s*\n*\s*STATEMENTS\s*\n*\s*$)" \
           r"|(\n*^\s*REPORT\s*\n*\s*OF\s*\n*\s*INDEPENDENT\s*\n*\s*(PUBLIC)?\s*\n*\s*ACCOUNTANTS\s*\n*\s*$)" \
           r"|(\n*^\s*Consolidated\s*\n*\s*(Statements|Results)\s*\n*\s*of\s*\n*\s*(Operations|Earnings|Income)\s*\n*\s*$)"


rf_start1 = r"(\n*^\s*item\s*\n*[1]\s*\n*\(?[A]\)?\s*\n*\s*[-‒–—.:\s]+\s*\n*R\s*\n*\s*I\s*\n*\s*S\s*\n*\s*K\s*\n*\s*)"
rf_end1 = r"(\n*^\s*item\s*\n*[1]\s*\n*\(?[B]\)?\s*\n*\s*[-‒–—.:\s]\s*\n*\s*)|(\n*^\s*item\s*\n*([2-9]|1[0-4])\s*\n*\(?[A-B]?\)?\s*\n*[-‒–—.:\s]+\s*\n*)"

rf_start2 = r"(\n*\s*item\s*\n*[1]\s*\n*\(?[A]\)?\s*\n*[-‒–—.:\s]\s*\n*\s*Risk\s*\n*\s*factors\s*\n*\s*)"
rf_end2 = r"(\n*\s*item\s*\n*([2-9]|1[0-4])\s*\n*\(?[A-B]?\)?\s*\n*[-‒–—.:\s]+\s*\n*)|(\n*^\s*item\s*\n*[1]\s*\n*\(?[B]\)?\s*\n*\s*[-‒–—.:\s]\s*\n*\s*)"

rf_start3 = r"(\n*^\s*RISK\s*\n*\s*Fators\s*\n*\s*)"
rf_end3 = r"(\n*^\s*Unresolved\s*\n*\s*Staff\s*\n*\s*Comments\s*\n*\s*)" \
           r"(\n*^\s*Properties\s*\n*\s*)" \
           r"(\n*^\s*legal\s*\n*\s*proceedings\s*\n*\s*)" \
           r"(\n*^\s*Mine\s*\n*\s*Safety\s*\n*\s*Disclosures\s*\n*\s*)" \
           r"(\n*^\s*Market\s*\n*\s*for\s*\n*\s*Registrant\s*\n*\s*.\s*\n*\s*s\s*\n*\s*Common\s*\n*\s*Equity\s*\n*\s*)" \
           r"(\n*^\s*Selected\s*\n*\s*Financial\s*\n*\s*Data\s*\n*\s*)" \
           r"(\n*^\s*MANAGEMENT.\s*\n*S\s*\n*DISCUSSION\s*\n*AND\s*\n*ANALYSIS\s*\n*)" \
           r"(\n*^\sQUANTITATIVE\s*\n*\s*AND\s*\n*\s*QUALITATIVE\s*\n*\s*DISCLOSURES\s*\n*\s*ABOUT\s*\n*\s*MARKET\s*\n*\s*RISK\s*\n*\s*$)"


qqdmr_start1 = r"(\n*^\s*item\s*\n*[7]\s*\n*\(?[A]\)?\s*\n*\s*[-‒–—.:\s]+.?\s*\n*\s*Q\s*\n*\s*U\s*\n*\s*A\s*\n*\s*N\s*\n*\s*T\s*\n*\s*I\s*\n*\s*T\s*\n*\s*A\s*\n*\s*T\s*\n*\s*I\s*\n*\s*V\s*\n*\s*E\s*\n*\s*)"
qqdmr_end1 = r"(\n*^\s*item\s*\n*([8-9]|1[0-4])\s*\n*\(?[A-B]?\)?\s*\n*[-‒–—.:\s]+\s*\n*)"

qqdmr_start2 = r"(\n*\s*item\s*\n*[7]\s*\n*\(?[A]\)?\s*\n*[-‒–—.:\s]+\s*\n*\s*Quantitative\s*\n*\s*and\s*\n*\s*Qualitatives*\n*\s*)"
qqdmr_end2 = r"(\n*\s*item\s*\n*([8-9]|1[0-4])\s*\n*\(?[A-B]?\)?\s*\n*[-‒–—.:\s]+\s*\n*)"

qqdmr_start3 = r"(\n*^\sQUANTITATIVE\s*\n*\s*AND\s*\n*\s*QUALITATIVE\s*\n*\s*DISCLOSURES\s*\n*\s*ABOUT\s*\n*\s*MARKET\s*\n*\s*RISK\s*\n*\s*$)"
qqdmr_end3 = r"|(\n*^\sFinancial\s*\n*\s*Statements\s*\n*\s*and\s*\n*\s*Supplementary\s*\n*\s*Data\s*\n*\s*$)" \
             r"|(\n*^\sChanges\s*\n*\s*in\s*\n*\s*and\s*\n*\s*Disagreements\s*\n*\s*$)" \
             r"|(\n*^\s*((Controls\s*\n*\s*and\s*\n*\s*Procedures$)|(Other\s*\n*\s*Information))\s*\n*\s*$)" \
             r"|(\n*^\s*Directors.?\s*\n*\s*Executive\s*\n*\s*Officers\s*\n*\s*$)" \
             r"|(\n*^\s*Executive\s*\n*\s*Compensation\s*\n*\s*$)" \
             r"|(\n*^\s*Security\s*\n*\s*Ownership\s*\n*\s*of\s*\n*\s*Certain\s*\n*\s*$)" \
             r"|(\n*^\s*Certain\s*\n*\s*Relationships\s*\n*\s*and\s*\n*\s*Related\s*\n*\s*$)" \
             r"|(\n*^\s*Principal\s*\n*\s*Accountant\s*\n*\s*Fees\s*\n*\s*$)"





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
        for match in re.finditer(mda_start1, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            start = match.start()
            #print(match)
            mend = re.search(mda_end1, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            #print(mend)
            output.append(text[start:start + mend.start()])


        if len(output)<1:
            for match in re.finditer(mda_start2, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                #print(match)
                mend = re.search(mda_end2, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend)
                output.append(text[start:start + mend.start()])

        if len(output)<1 or len(word_tokenize(max(output, key=len)))<200:
            for match in re.finditer(mda_start3, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                #print(match)
                mend = re.search(mda_end3, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                #print(mend)
                output.append(text[start:start + mend.start()])

        words = word_tokenize(max(output, key=len) )
        words = [word.lower() for word in words if word.isalpha()]
        words = [word.lower() for word in words if not word.upper() in stop_words]

        wc = len(words)
        cwc = 0
        for word in words:
            if syllable_count(word) > 2:
                cwc += 1
        return [wc,cwc]

    except Exception as e:
        print(e)
        return [0,0]





def rf_extract(text):

    try:
        output = []

        for match in re.finditer(rf_start1, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            start = match.start()
            mend = re.search(rf_end1, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            output.append(text[start:start + mend.start()])

        if len(output)<1:
            for match in re.finditer(rf_start2, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                mend = re.search(rf_end2, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                output.append(text[start:start + mend.start()])

        if len(output)<1 :
            for match in re.finditer(rf_start3, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                mend = re.search(rf_end3, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                output.append(text[start:start + mend.start()])

        words = word_tokenize(max(output, key=len))
        words = [word.lower() for word in words if word.isalpha()]
        words = [word.lower() for word in words if not word.upper() in stop_words]

        wc = len(words)
        cwc = 0
        for word in words:
            if syllable_count(word) > 2:
                cwc += 1
        return[wc,cwc]
    except:
        return[0,0]





def qqdmr_extract(text):

    try:
        output = []

        for match in re.finditer(qqdmr_start1, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            start = match.start()
            mend = re.search(qqdmr_end1, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            output.append(text[start:start + mend.start()])

        if len(output)<1:
            for match in re.finditer(qqdmr_start2, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                mend = re.search(qqdmr_end2, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                output.append(text[start:start + mend.start()])

        if len(output)<1:
            for match in re.finditer(qqdmr_start3, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
                start = match.start()
                mend = re.search(qqdmr_end3, text[start:], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                output.append(text[start:start + mend.start()])

        words = word_tokenize(max(output, key=len))
        words = [word.lower() for word in words if word.isalpha()]
        words = [word.lower() for word in words if not word.upper() in stop_words]

        wc = len(words)
        cwc = 0
        for word in words:
            if syllable_count(word) > 2:
                cwc += 1
        return [wc,cwc]
    except:
        return [0,0]



