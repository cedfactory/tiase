from abc import ABCMeta, abstractmethod

class Classifier(object):
    __metaclass__=ABCMeta
    @abstractmethod
    def build_model():
        pass

    @abstractmethod
    def create_model():
        pass

    @abstractmethod
    def get_analysis():
        pass


