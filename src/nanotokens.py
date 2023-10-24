import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

class NanoTokens():
    """
    This class handles the tokenisation, encoding and decoding of text.
    token_method (str) - can be 1-character, 2-character
    text (str) - the input text to train on

    example use:
    nanotokens = NanoTokens(token_method='n-character', text=text) # creates the NanoTokens object
    encoded_text = nanotokens.encode(text) # encodes the text using the token method
    decoded_text = nanotokens.decode(encoded_text) # decodes the text
    """

    def __init__(self, token_method: str = 'n-character', n: int = 1, text: str = ''):
        self.token_method = token_method
        self.n = n
        self.text = text
        self.tokens = self.get_tokens()
        self.vocab_size = len(self.tokens)
        self.ttoi = { token:index for index,token in enumerate(self.tokens) } # Converts a token to integer
        self.itot = { index:token for index,token in enumerate(self.tokens) } # Converts an integer to token

    def get_tokens(self):
        """
        returns:
            a list of all the tokens within a text, using a method
        """
        # Using characters as our tokens isn't a bad idea, but it might be smarter to consider 3 characters at a time etc
        # This is because we could represent our sentence in a much smaller list, just with integers up to 5000 instead of 65 for example
        # There's a obviously an optimal combination, but to keep things simple we're going to use a 1/2-character tokenizer

        if (self.token_method == 'n-character'):
            
            chars_1gram = sorted(list(set(self.text))) # the set of every unique character in the text
            if (self.n == 1): return chars_1gram
            chars_ngram = sorted(list(set(re.findall('[a-zA-Z]{'+str(self.n)+'}',self.text)))) # the set of every n alpha pairs
            # we add the 1 grams to the n grams so that we can still predict words like 'I' or 'A'
            # and include spaces, symbols, numbers, etc...
            chars_ngram += chars_1gram 
            return chars_ngram
        
        elif (self.token_method == 'n-gram'):
            ngram = ngrams(word_tokenize(self.text), self.n)
            return list(ngram)

    def encode(self, input_text: str = ''):
        """
        given a valid token_method, encode the text with that method
        by mapping the index to the characters
        returns:
            a list of the input text tokenised using the token_method
        """

        if (self.token_method == 'n-character'):
            encoded_text = []
            i = 0
            max_i = len(input_text)-self.n+1
            while (i < max_i):
                # Get character token of length n
                current_token = ''.join([input_text[i+j] for j in range(self.n)])
                if (current_token in self.tokens):
                    # for every n characters, if it's a recognisable token
                    # encode it!
                    encoded_text.append(self.ttoi[current_token])
                    i+=self.n
                    # increase i by n
                else:
                    # else it must be a single char token
                    encoded_text.append(self.ttoi[input_text[i]])
                    i+=1
            return encoded_text
        
        elif (self.token_method == 'n-gram'):
            input_text = word_tokenize(input_text)
            encoded_text = []
            i = 0
            max_i = len(input_text)-self.n+1
            while (i < max_i):
                # Get word token of length n
                current_token = tuple([input_text[i+j] for j in range(self.n)])

                if (current_token in self.tokens):
                    # for every n words, if it's a recognisable token
                    # encode it!
                    encoded_text.append(self.ttoi[current_token])
                    i+=self.n
                    # increase i by n
                   
            return encoded_text
        
    def decode(self, input_list: list = []):
        """
        given a valid list of tokens
        returns:
            a decoded string of those tokens
        """
        if (self.token_method == 'n-character'):
            return ''.join([self.itot[index] for index in input_list])
        elif (self.token_method == 'n-gram'):
            decoded_string = ''
            for i in input_list:
                print(self.itot[index] for index in input_list)
            return decoded_string
        