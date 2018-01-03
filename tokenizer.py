from nltk import word_tokenize
from tqdm import tqdm
from csv import DictReader
from csv import DictWriter

count = 0
with open("Message-11-23-17.csv") as infile:
	with open("tokenized_messages.csv", "w") as outfile:
		reader = DictReader(infile)
		fieldnames = reader.fieldnames+["Tokens"]
		print(fieldnames)
		writer = DictWriter(outfile, fieldnames=fieldnames)


		writer.writeheader()
		for line in tqdm(reader):
			token_string = ""
			if line["Content"] != None:
				token_string += " ".join(word_tokenize(line["Content"].lower()))
			line["Tokens"] = token_string
			if None in line:
				del line[None]
			writer.writerow(line)


            
            