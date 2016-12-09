from pprint import pprint as pp

from deepopt.deepopt import DeepOptEpoch
from deepopt.chooser import get_max_epoch
from .family.simple import SimpleHybridFamily
from chainer.training import extensions
import chainer

class Collection(object):
    def __init__(self, name, path="_models", input_dims=1, nepochs=10, verbose=False):
        self.name = name
        self.path = path
        self.folder = '{}/{}'.format(self.path,name)
        self.verbose = verbose
        self.input_dims = input_dims

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

    def set_model_family(self, family, **kwargs):
        self.family = family(folder=self.folder, input_dims=self.input_dims, **kwargs)

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

            trainer, model, chain = self.family.train_model(self.trainset, self.testset, **point)
            
            # re-evaluate the result, TODO: reeval from previous epoch as well
            result = trainer.evaluate()
            meta = {}
            for k,v in result.iteritems():
                if hasattr(v,'tolist'):
                    v = v.tolist()
                meta['validation/'+k] = v
            for k,v in point.iteritems():
                 meta[k] = str(v)
            meta['epoch'] = point['nepochs']
            meta['validation/numtrain'] = len(self.trainset)
            meta['validation/numtest'] = len(self.testset)
            main_report = getattr(self.family, 'main_report', "validation/main/accuracy")
            do.add_point(point['nepochs'], meta[main_report], meta=meta, **point)
                
            #meta_reports = getattr(self.family, 'meta_reports', [])
            #metas = {}
            #for meta in meta_reports:
            #    metas[meta] = trainer.get_log_result(meta)
            #    print(meta,metas[meta][-1])
            #main_report = getattr(self.family, 'main_report', "validation/main/accuracy")
            #do.add_points(range(1, int(point['nepochs'])+1), trainer.get_log_result(main_report), metas=metas, **point)

        do.fit()

        # Train
        while i < niters:
            #print(i)
            point = self.do.sample_point()
            i += max(1, point['nepochs'] - get_max_epoch(do, point))
            trainer, model = self.family.train_model(self.trainset, self.testset, **point)
            
            result = trainer.evaluate()
            meta = {}
            for k,v in result.iteritems():
                if hasattr(v,'tolist'):
                    v = v.tolist()
                meta['validation/'+k] = v
            for k,v in point.iteritems():
                 meta[k] = str(v)
            meta['epoch'] = point['nepochs']
            meta['validation/numtrain'] = len(self.trainset)
            meta['validation/numtest'] = len(self.testset)
            main_report = getattr(self.family, 'main_report', "validation/main/accuracy")
            do.add_point(point['nepochs'], meta[main_report], meta=meta, **point)
            
            #meta_reports = getattr(self.family, 'meta_reports', [])
            #meta_reports = [
            # 'epoch',
            # 'validation/main/branch0accuracy',
            # 'validation/main/branch1accuracy',
            # 'validation/main/branch2accuracy',
            # 'validation/main/branch3accuracy',
            # 'validation/main/branch4accuracy',
            # 'validation/main/branch5accuracy',
            # 'validation/main/branch6accuracy',
            # 'validation/main/branch7accuracy',
            # 'validation/main/branch8accuracy',
            # 'validation/main/accuracy',
            # 'validation/main/communication0',
            # 'validation/main/communication1',
            # 'validation/main/memory',
            # 'validation/main/ent_T',
            # 'validation/main/branch0exit',
            # 'validation/main/branch1exit',
            # 'validation/main/branch2exit'
            #]
            #
            #metas = {}
            #for meta in meta_reports:
            #    metas[meta] = trainer.get_log_result(meta)
            #main_report = getattr(self.family, 'main_report', "validation/main/accuracy")
            #
            #do.add_points(range(1, int(point['nepochs'])+1), trainer.get_log_result(main_report), metas=metas, **point)
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
