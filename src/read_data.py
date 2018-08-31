import re

def read_trump_speeches(file_loc):
    # Read file
    file = open(file_loc, 'r',encoding='utf-8-sig')
    speeches = file.read()

    # Remove text between brackets, such as (inaudible) or (laughter)
    speeches = re.sub("[\(\[].*?[\)\]]", "", speeches)
    # Remove the speech introductions
    speeches = re.sub(r'SPEECH.+?\n', '', speeches)
    # Replace multiple periods with a single one.
    speeches = re.sub('\.+','. ',speeches)
    # different uses of this character
    speeches = re.sub('\'','’',speeches)
    # Replace new lines with spaces
    speeches = re.sub('\n',' ', speeches)

    # Treat the following interpunction characters as separate words, so we can generate them.
    speeches = re.sub('\. ',' . ', speeches)
    speeches = re.sub(', ',' , ', speeches)
    speeches = re.sub('\? ',' ? ', speeches)
    speeches = re.sub('! ',' ! ', speeches)
    speeches = re.sub('; ',' ; ', speeches)
    punc = '.,?!;'

    # Keep only this set of characters, replace multiple whitespace with single, and convert to lower case.
    speeches = re.sub('[^0-9a-zA-Z\.,\?!;’]+', ' ', speeches)
    speeches = re.sub('\s+',' ', speeches)
    speeches = speeches.lower()
    speeches = speeches.split(' ')
    return speeches