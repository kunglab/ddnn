from pprint import pprint as pp

from deepopt.deepopt import DeepOptEpoch
from deepopt.chooser import get_max_epoch
from .family.simple import SimpleHybridFamily

class Collection(object):
    def __init__(self, name, path="_models", nepochs=10, verbose=False):
        self.name = name
        self.path = path
        self.folder = '{}/{}'.format(self.path,name)
        self.verbose = verbose

        self.searchspace = None
        self.set_model_family(SimpleHybridFamily)
        self.set_nepochs(nepochs)

    def set_nepochs(self, nepochs):
        self.nepochs = nepochs
        self.do = DeepOptEpoch(nepochs=nepochs, folder=self.folder)
        if self.searchspace is not None:
            self.set_searchspace(self.searchspace)

    def add_trainset(self, trainset):
        self.trainset = trainset

    def add_testset(self, testset):
        self.testset = testset

    def set_chooser(self, chooser):
        self.do.set_chooser(chooser)

    def set_model_family(self, family):
        self.family = family(folder=self.folder)

    def set_searchspace(self, **searchspace):
        self.searchspace = searchspace
        for k,v in searchspace.iteritems():
            self.do.add_param(k, v)

    def set_constraints(self, constraintfn):
        self.do.set_constraints(constraintfn)

    def print_status(self):
        trace = self.do.get_traces()[-1]
        sample = dict(zip(self.do.params, trace['x']))
        print('Acc: {:2.3f}'.format(trace['y']))
        pp(sample)
        print('')

    def train(self, niters=10, bootstrap_nepochs=2):
        do = self.do
        i = 0
        # Bootstrap epochs
        for point in do.get_bootstrap_points(bootstrap_nepochs):
            i += bootstrap_nepochs
            #print(i)
            if self.verbose:
                print('Bootstrap: {}'.format(point))

            trainer, model = self.family.train_model(self.trainset, self.testset, **point)
            metas = dict(branch0accuracy=trainer.get_log_result("validation/main/branch0accuracy"),
                         branch1accuracy=trainer.get_log_result("validation/main/branch1accuracy"),
                         branch2accuracy=trainer.get_log_result("validation/main/branch2accuracy"),
                         branch0exit=trainer.get_log_result("validation/main/branch0exit"),
                         branch1exit=trainer.get_log_result("validation/main/branch1exit"),
                         branch2exit=trainer.get_log_result("validation/main/branch2exit"),
                         numsamples=trainer.get_log_result("validation/main/numsamples"))
            do.add_points(range(1, int(point['nepochs'])+1), trainer.get_log_result("validation/main/accuracy"), metas=metas, **point)

        do.fit()

        # Train
        while i < niters:
            print(i)
            point = self.do.sample_point()
            i += max(1, point['nepochs'] - get_max_epoch(do, point))
            trainer, model = self.family.train_model(self.trainset, self.testset, **point)
            metas = dict(branch0accuracy=trainer.get_log_result("validation/main/branch0accuracy"),
                         branch1accuracy=trainer.get_log_result("validation/main/branch1accuracy"),
                         branch2accuracy=trainer.get_log_result("validation/main/branch2accuracy"),
                         branch0exit=trainer.get_log_result("validation/main/branch0exit"),
                         branch1exit=trainer.get_log_result("validation/main/branch1exit"),
                         branch2exit=trainer.get_log_result("validation/main/branch2exit"),
                         numsamples=trainer.get_log_result("validation/main/numsamples"))
            do.add_points(range(1, int(point['nepochs'])+1), trainer.get_log_result("validation/main/accuracy"), metas=metas, **point)
            do.fit()

            if self.verbose:
                self.print_status()

        # Get the best model
        point = do.get_best_point()
        chain, model = self.family.load_chain_model(**point)

        # Associate with this collection
        self.model = model
        self.chain = chain
        return self.get_do_traces()

    def get_do_traces(self):
        return self.do.get_traces()

    def generate_c(self, in_shape):
        return self.model.generate_c(in_shape)

    def predict(self, x):
        return self.model(x)

    def generate_container(self):
        raise Exception("Not Implemented")
        #return self.model.generate_container_zip()
