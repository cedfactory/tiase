from sklearn.ensemble import VotingClassifier

def PrepareModelsForMetaClassifierVoting(models):
    formatted_models = []
    return formatted_models


class MetaClassifierVoting:
    def __init__(self, models, params):
        '''
        format for models : [('model1_name', model1), ('model2_name', model2)]
        '''
        self.models = models

        self.seq_len = 21
        if params:
            self.seq_len = params.get("seq_len", self.seq_len)

    def build(self):
        self.meta_model = VotingClassifier(estimators=models, voting='hard')
