import logging
from sklearn.metrics import log_loss
import re
import spacy
nlp = spacy.load('en_core_web_md')

SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}


def loss(*args):
    return log_loss(*args)


def clean_string(text):

    def pad_str(s):
        return ' '+s+' '

    # Empty question

    if type(text) != str or text == '':
        return ''

    text = pad_str(text.lower())

    def lower_first_char(pattern):
        matched_string = pattern.group(0)
        return matched_string[:-1] + matched_string[-1].lower()

    text = re.sub("(?<=[\.\?\)\!\'\"])[\s]*.", lower_first_char, text)

    # Replace weird chars in text
    text = re.sub("’", "'", text)  # special single quote
    text = re.sub("`", "'", text)  # special single quote
    text = re.sub("“", '"', text)  # special double quote
    text = re.sub("？", "?", text)
    text = re.sub("…", " ", text)
    text = re.sub("é", "e", text)
    text = re.sub("\?", "", text)
    text = re.sub("\,", " ", text)

    # Clean shorthands

    # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable,
    # I choose to compromise are kill "'s" directly
    text = re.sub("\'s", " ", text)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    # better regex provided by @armamut
    text = re.sub(r"(\W|^)([0-9]+)[kK](\W|$)", r"\1\g<2>000\3", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    # replace the float numbers with a random number, it will be parsed as number afterward,
    # and also been replaced with word "number"

    # text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)

    # remove anything between numbers, i.e. 15x000 -> 15000
    #
    text = re.sub('(?<=[0-9])[^0-9](?=[0-9])', "", text)

#     # all numbers should separate from words, this is too aggressive

#     def pad_number(pattern):
#         matched_string = pattern.group(0)
#         return pad_str(matched_string)
#     text = re.sub('[0-9]+', pad_number, text)

    # add padding to punctuations and special chars, we still need them later

    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)

    def pad_pattern(pattern):
        matched_string = pattern.group(0)
        return pad_str(matched_string)
    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text)

    # replace non-ascii word with special word
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text)

    # indian dollar

    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text

    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
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

    # typos identified with my eyes

    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" demoniti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" demoneti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    # not typo, but GloVe can't find it
    text = re.sub(r" undergraduation ", " undergraduate ", text)
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)
    text = re.sub(r" begineer ", " beginner ", text)
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)
    text = re.sub(r" litrate ", " literate ", text)

    # for words like A-B-C-D or "A B C D",
    # if A,B,C,D individuaally has vector in glove:
    #     it can be treat as separate words
    # else:
    #     replace it as a special word, A_B_C_D is enough, we'll deal with that word later
    #
    # Testcase: 'a 3-year-old 4 -tier car'

    def dash_dealer(pattern):
        matched_string = pattern.group(0)
        splited = matched_string.split('-')
        splited = [sp.strip() for sp in splited if sp != ' ' and sp != '']
        joined = ' '.join(splited)
        parsed = nlp(joined)
        for token in parsed:
            # if one of the token is not common word, then join the word into one single word
            if not token.has_vector or token.text in SPECIAL_TOKENS.values():
                return '_'.join(splited)
        # if all tokens are common words, then split them
        return joined

    text = re.sub("[a-zA-Z0-9\-]*-[a-zA-Z0-9\-]*", dash_dealer, text)

    # try to see if sentence between quotes is meaningful
    # rule:
    #     if exist at least one word is "not number" and "length longer than 2" and
    #          "it can be identified by SpaCy":
    #         then consider the string is meaningful
    #     else:
    #         replace the string with a special word, i.e. quoted_item
    # Testcase:
    # i am a good (programmer)      -> i am a good programmer
    # i am a good (programmererer)  -> i am a good quoted_item
    # i am "i am a"                 -> i am quoted_item
    # i am "i am a programmer"      -> i am i am a programmer
    # i am "i am a programmererer"  -> i am quoted_item

    def quoted_string_parser(pattern):
        string = pattern.group(0)
        parsed = nlp(string[1:-1])
        is_meaningful = False
        for token in parsed:
            # if one of the token is meaningful, we'll consider the full string is meaningful
            if len(token.text) > 2 and not token.text.isdigit() and token.has_vector:
                is_meaningful = True
            elif token.text in SPECIAL_TOKENS.values():
                is_meaningful = True

        if is_meaningful:
            return string
        else:
            return pad_str(string[0]) + SPECIAL_TOKENS['quoted'] + pad_str(string[-1])

    text = re.sub('\".*\"', quoted_string_parser, text)
    text = re.sub("\'.*\'", quoted_string_parser, text)
    text = re.sub("\(.*\)", quoted_string_parser, text)
    text = re.sub("\[.*\]", quoted_string_parser, text)
    text = re.sub("\{.*\}", quoted_string_parser, text)
    text = re.sub("\<.*\>", quoted_string_parser, text)

    text = re.sub('[\(\)\[\]\{\}\<\>\'\"]', pad_pattern, text)

    # the single 's' in this stage is 99% of not clean text, just kill it
    text = re.sub(' s ', " ", text)

    # reduce extra spaces into single spaces
    text = re.sub('[\s]+', " ", text)
    text = text.strip()

    return text.lower()


done = 0


def normalize(text):
    global done
    done += 1
    if not done % 100000:
        logging.info(done)
    text = clean_string(text)

    return text.split()


if __name__ == "__main__":
    print(clean_string("I'm well-prepared, vrd-ddw d   deef?"))
