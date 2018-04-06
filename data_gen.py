import re

class Corpus(object):
    """
    """
    def __init__(self,in_file,
        target_file=None):
        self.in_file = in_file
        self.target_file = target_file
        self.__iter__()


    def __iter__(self):
        for i,(line,target_list) in enumerate(zip(open(self.in_file),open(self.target_file))):
            line = re.sub('eos','',line)
            yield ' '.join(line.strip().replace('-',' ').split(',')),target_list.strip().split(',')


class hierarchicalCorpus(object):
    """
    """
    def __init__(self,in_file,
        target_file=None):
        self.in_file = in_file
        self.target_file = target_file
        self.__iter__()


    def __iter__(self):
        for i,(line,target_list) in enumerate(zip(open(self.in_file),open(self.target_file))):
            sentences = line.rstrip().replace('-',' ').split('eos')[:-1]
            sentences = [sent.split(',') for sent in sentences]
            sentences = [' '.join(sent) for sent in sentences]

            yield sentences,target_list.strip().split(',')