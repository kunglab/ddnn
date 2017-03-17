from chainer_sequential.chain import Chain
from deepopt.trainer import Trainer
import chainer
import chainer.serializers as S
from chainer_sequential.sequential import Sequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F

class FloatFamily:
    def __init__(self, folder="_models/float", prefix=None, input_dims=1, output_dims=10):
        self.folder = folder
        self.prefix = prefix
        self.input_dims = input_dims
        self.output_dims = output_dims

    def get_configurable_params(self):
        return ["nfilters_embeded", "nlayers_embeded", "lr"]

    def generate_model(self, **kwargs):
        nfilters_embeded = int(kwargs.get("nfilters_embeded", 1))
        nlayers_embeded = int(kwargs.get("nlayers_embeded", 1))

        model = Sequential()
        for i in range(nlayers_embeded):
            if i == 0:
                nfilters = self.input_dims
            else:
                nfilters = nfilters_embeded
            model.add(Convolution2D(nfilters, nfilters_embeded, 3, 1, 1))
            model.add(max_pooling_2d(3,1,1))
            model.add(BatchNormalization(nfilters_embeded))
            model.add(Activation('bst'))

        model.add(Linear(None, self.output_dims))

        model.build()
        return model

    def load_chain_model(self, **kwargs):
        name = self.get_name(**kwargs)
        path = '{}/{}'.format(self.folder,name)
        epoch = int(kwargs.get("nepochs",2))
        fn = "{}/chain_snapshot_epoch_{:06}".format(path,epoch)

        chain, model = self.setup_chain_model(**kwargs)
        S.load_npz(fn, chain)
        return chain, model

    def setup_chain_model(self, **kwargs):
        model = self.generate_model(**kwargs)

        branchweight = kwargs.get("branchweight", 3)
        lr = kwargs.get("lr", 0.001)

        chain = Chain(branchweight=branchweight)
        chain.add_sequence(model)
        chain.setup_optimizers('adam', lr)
        return chain, model

    def get_name(self, **kwargs):
        if self.prefix is not None:
            name = "{}_".format(self.prefix)
        else:
            name = ""
        for k,v in kwargs.iteritems():
            if k=='nepochs' or k=='ent_T':
                continue
            name = "{}_{}_{}".format(name, k, v)
        return name

    def train_model(self, trainset, testset, **kwargs):
        chain, model = self.setup_chain_model(**kwargs)

        nepochs = int(kwargs.get("nepochs", 2))
        name = self.get_name(**kwargs)

        trainer = Trainer('{}/{}'.format(self.folder,name), chain, trainset,
                          testset, nepoch=nepochs, resume=True)
        trainer.run()

        return trainer, model, chain
