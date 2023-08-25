from transformers import AutoTokenizer

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# See what the vocabulary contains.  each entry is a 
# pair of token and ID. For example, we can represent the 
# word optional with the number 11902. You might have noticed a 
# special character, Ġ, preceding certain tokens. This character 
# represents a space
print(tokenizer.vocab)

# use the tokenizer object to convert a sentence into tokens and IDs.
token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")

# The .encode() method can convert any given text into a numerical representation, 
# a list of integers. To further investigate the process, we can use the 
# .convert_ids_to_tokens() function that shows the extracted tokens. As an example, 
# you can observe that the word "tokenizer" has been split into a combination of 
# "token" + "izer" tokens.

print("Tokens:   ", tokenizer.convert_ids_to_tokens(token_ids))
print("Token IDs:", token_ids)
