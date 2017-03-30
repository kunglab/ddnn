from pprint import pprint as pp
import os

from deepopt.deepopt import DeepOptEpoch
from deepopt.chooser import get_max_epoch
from .family.simple import SimpleHybridFamily
from chainer_sequential.binary.utils import binary_util as bu
from chainer.training import extensions
import chainer

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class Collection(object):
    def __init__(self, name="collection", path="_models", nepochs=10, verbose=False):
        # create the folder if it does not exist
        if not os.path.exists("_models"):
            mkdir_p(path)

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

    def set_model_family(self, family, **kwargs):
        self.family = family(folder=self.folder, **kwargs)

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

    def train(self, niters=1, bootstrap_nepochs=1):
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

            if result.get('main/branch0exit') is not None and point.get("target_branch0exit") is not None:
                while result['main/branch0exit'] < point["target_branch0exit"]:
                    if self.verbose:
                        print("trying at exit %", result['main/branch0exit'], chain.ent_Ts[0])
                    chain.ent_Ts[0] += 0.01
                    result = trainer.evaluate()
                    result['ent_T'] = chain.ent_Ts[0]

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
            trainer, model, chain = self.family.train_model(self.trainset, self.testset, **point)

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

    def generate_c_old(self, in_shape, **kwargs):
        return self.model.generate_c(in_shape, **kwargs)

    def generate_c(self, save_file, in_shape):
        c_code = self.model.generate_c(in_shape)
        save_dir = os.path.join(os.path.split(save_file)[:-1])[0]
        if not os.path.exists(save_dir) and save_dir != '':
            os.makedirs(save_dir)

        with open(save_file, 'w+') as fp:
            fp.write(c_code)

    def generate_inter_results_c(self, save_file, x):
        _, inter = self.model(x, test=True, output_inter=True)
        res = ''
        res += bu.np_to_floatC(inter[0][0,0], 'x_in', 'row_major') + '\n'

        for i in range(len(inter[1:-1])):
            inter_res = inter[i+1]
            # inter_res = inter_res.reshape(-1, inter_res.shape[-1])
            inter_res = inter_res.reshape(1, -1)
            res += bu.np_to_uint8C(bu.binarize_real(inter_res),
                                   'inter' + str(i+1), 'row_major')
            res += '\n'

        res += bu.np_to_floatC(inter[-1], 'y_out', 'row_major') + '\n'

        save_dir = os.path.join(os.path.split(save_file)[:-1])[0]
        if not os.path.exists(save_dir) and save_dir != '':
            os.makedirs(save_dir)

        with open(save_file, 'w+') as fp:
            fp.write(res)

        return inter

    def predict(self, x):
        return self.model(x)

    def generate_container(self):
        raise Exception("Not Implemented")
        #return self.model.generate_container_zip()
