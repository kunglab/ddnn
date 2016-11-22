from deepopt.deepopt import DeepOptEpoch

class SimpleHybridFamily:
    def generate_model(self, **kwargs):
        # TODO
        return model
    def load_model(self, **kwargs):
        # TODO
        return
    def train_model(self, trainset, testset, **kwargs):
        # TODO
        return trainer, model

class Collection:
    def __init__(self, name):
        self.name = name
        self.set_model_family(SimpleHybridFamily())
        self.do = DeepOptEpoch()
    def add_training_data(self, trainset):
        self.trainset = trainset
    def add_validation_data(self, testset):
        self.testset = testset
    def set_model_family(self, family):
        self.family = family
    def set_constraints(self, **constraints):
        for k,v in constraints.iteritems():
            self.do.add_param(k, v)
    def train(self, niters=10, bootstrap_nepochs=2):
        # Bootstrap epochs
        for point in self.deepopt.get_bootstrap_points(bootstrap_nepochs):
            trainer, model = self.family.train(self.trainset, self.testset, **point)
            do.add_points(range(1,point['nepochs']+1), trainer.get_log_result(), **point)
            
        self.deepopt.start_traces()
        # Train
        for i in range(niters):
            point = self.deepopt.sample_point()
            trainer, model = self.family.train(self.trainset, self.testset, **point)
            do.add_points(range(1,point['nepochs']+1), trainer.get_log_result(), **point)
        traces = self.deepopt.stop_traces()
        
        # Get the best model
        point = do.get_best_point()
        model = self.family.load_model(**point)
        
        # Associate with this collection
        self.model = model
        return traces
    def generate_c(self, in_shape):
        return self.model.generate_c(in_shape)
    def predict(self, x):
        return self.model(x)
    def generate_container_zip(self):
        raise Exception("Not Implemented")
        #return self.model.generate_container_zip()