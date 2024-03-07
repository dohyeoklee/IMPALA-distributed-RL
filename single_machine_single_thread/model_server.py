from network import Policy

class ModelServer():
    def __init__(self,variants):
        self.model = Policy(variants['state_size'],variants['action_size'],variants['hidden_size'])

    def set_model_weights(self,weights):
        self.model.load_state_dict(weights)

    def load_model_weights(self,model):
        model.load_state_dict(self.model.state_dict())