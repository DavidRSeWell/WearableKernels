
class TransferKernel:

    def __init__(self,data,pre_model=None,set_points=1000):
        self.data = data
        self.pre_model = pre_model
        self.set_points = set_points

    def create_nlp_vocab(self):
        """
        Create a dictionary of "word: vector"
        :return:
        """

