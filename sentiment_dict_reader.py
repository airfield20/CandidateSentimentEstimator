import sys

class SentimentDictReader:
    def __init__(self, fname):
        try:
            with open(fname,"r",errors='ignore') as file:
                self.words = file.readlines()
        except IOError as e:
            sys.exit("ERROR: " + fname + e)
        for index,line in enumerate(self.words):
            if line[0] != ';':
                break
        self.words = self.words[index+1:]
        for index,line in enumerate(self.words):
            self.words[index] = line.strip()


if __name__ == '__main__':
    reader = SentimentDictReader("dictionaries/positive-words.txt")
    x=0;