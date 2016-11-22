from deepopt.deepopt import DeepOptEpoch
from .family.simple import SimpleHybridFamily

class Collection:
    def __init__(self, name, nepochs=10):
        self.name = name
        self.set_model_family(SimpleHybridFamily(prefix=self.name))
        self.do = DeepOptEpoch(nepochs=nepochs)
    def set_chooser(self, chooser):
        self.do.set_chooser(chooser)        
    def add_trainset(self, trainset):
        self.trainset = trainset
    def add_testset(self, testset):
        self.testset = testset
    def set_model_family(self, family):
        self.family = family
    def set_searchspace(self, **searchspace):
        for k,v in searchspace.iteritems():
            self.do.add_param(k, v)
    def set_constraints(self, constraintfn):
        self.do.set_constraints(constraintfn)
    def train(self, niters=10, bootstrap_nepochs=2):
        do = self.do
        # Bootstrap epochs
        for point in do.get_bootstrap_points(bootstrap_nepochs):
            trainer, model = self.family.train_model(self.trainset, self.testset, **point)
            do.add_points(range(1,int(point['nepochs'])+1), trainer.get_log_result("validation/main/accuracy"), **point)
            
        do.fit()
        do.start_traces()
        # Train
        for i in range(niters):
            point = self.do.sample_point()
            trainer, model = self.family.train_model(self.trainset, self.testset, **point)
            do.add_points(range(1,int(point['nepochs'])+1), trainer.get_log_result("validation/main/accuracy"), **point)
            do.fit()
            
        traces = do.stop_traces()
        
        # Get the best model
        point = do.get_best_point()
        chain, model = self.family.load_chain_model(**point)
        
        # Associate with this collection
        self.model = model
        self.chain = chain
        return traces
    def generate_c(self, in_shape):
        return self.model.generate_c(in_shape)
    def predict(self, x):
        return self.model(x)
    def generate_container(self):
        raise Exception("Not Implemented")
        #return self.model.generate_container_zip()