import re

class NanoTokens():
    """
    This class handles the tokenisation, encoding and decoding of text.
    token_method (str) - can be 1-character, 2-character
    text (str) - the input text to train on

    example use:
    nanotokens = NanoTokens(token_method='2-character', text=text) # creates the NanoTokens object
    encoded_text = nanotokens.encode(text) # encodes the text using the token method
    decoded_text = nanotokens.decode(encoded_text) # decodes the text
    """

    def __init__(self, token_method: str = '2-character', text: str = ''):
        self.token_method = token_method
        self.text = text
        self.tokens = self.get_tokens()
        self.vocab_size = len(self.tokens)
        self.stoi = { character:index for index,character in enumerate(self.tokens) } # Converts a string to integer
        self.itos = { index:character for index,character in enumerate(self.tokens) } # Converts an integer to string

    def get_tokens(self):
        """
        returns:
            a list of all the tokens within a text, using a method
        """
        # Using characters as our tokens isn't a bad idea, but it might be smarter to consider 3 characters at a time etc
        # This is because we could represent our sentence in a much smaller list, just with integers up to 5000 instead of 65 for example
        # There's a obviously an optimal combination, but to keep things simple we're going to use a 1/2-character tokenizer

        if (self.token_method == '3-character'):
            
            chars_1gram = sorted(list(set(self.text))) # the set of every unique character in the text
            chars_3gram = sorted(list(set(re.findall('[a-zA-Z]{3}',self.text)))) # the set of every 2 alpha pairs
            # we add the 1 grams to the 2 grams so that we can still predict words like 'I' or 'A'
            # and include spaces, symbols, numbers, etc...
            chars_3gram += chars_1gram 
            return chars_3gram
        
        elif (self.token_method == '2-character'):
            
            chars_1gram = sorted(list(set(self.text))) # the set of every unique character in the text
            chars_2gram = sorted(list(set(re.findall('[a-zA-Z]{2}',self.text)))) # the set of every 2 alpha pairs
            # we add the 1 grams to the 2 grams so that we can still predict words like 'I' or 'A'
            # and include spaces, symbols, numbers, etc...
            chars_2gram += chars_1gram 
            return chars_2gram
        
        elif (self.token_method == '1-character'):
            return sorted(list(set(self.text))) # the list of every unique character in the text
        
    def encode(self, input_text: str = ''):
        """
        given a valid token_method, encode the text with that method
        by mapping the index to the characters
        returns:
            a list of the input text tokenised using the token_method
        """

        if (self.token_method == '3-character'):
            encoded_text = []
            i = 0
            while (i < len(input_text)-2):
                if (input_text[i]+input_text[i+1]+input_text[i+2] in self.tokens): 
                    # for every 2 characters, if it's a recognisable token
                    # encode it!
                    encoded_text.append(self.stoi[input_text[i]+input_text[i+1]+input_text[i+2]])
                    i+=3
                else:
                    # else it must be a single char token
                    encoded_text.append(self.stoi[input_text[i]])
                    i+=1
            return encoded_text
        
        elif (self.token_method == '2-character'):
            encoded_text = []
            i = 0
            while (i < len(input_text)-1):
                if (input_text[i]+input_text[i+1] in self.tokens): 
                    # for every 2 characters, if it's a recognisable token
                    # encode it!
                    encoded_text.append(self.stoi[input_text[i]+input_text[i+1]])
                    i+=2
                else:
                    # else it must be a single char token
                    encoded_text.append(self.stoi[input_text[i]])
                    i+=1
            return encoded_text
        
        elif (self.token_method == '1-character'):
            return [self.stoi[character] for character in input_text]  # Convert a string to a list of integers

    def decode(self, input_list: list = []):
        """
        given a valid list of tokens
        returns:
            a decoded string of those tokens
        """
        return ''.join([self.itos[index] for index in input_list])
        