from chainer_sequential.multiinputchain import Chain
from deepopt.trainer import Trainer
import chainer
import chainer.serializers as S
from chainer_sequential.sequential import Sequential
from chainer_sequential.multiinputsequential import MultiInputSequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F

class EBNNFamily:
    def __init__(self, folder="_models/ebnn", prefix=None, input_dims=3, output_dims=3, batchsize=10):
        self.folder = folder
        self.prefix = prefix
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batchsize = batchsize

    def get_configurable_params(self):
        return ["nfilters_embeded", "nlayers_embeded", "nfilters_embeded_last", "branchweight", "lr", "ent_T"]

    def generate_model(self, **kwargs):
        nfilters_embeded_last = int(kwargs.get("nfilters_embeded_last", 1))
        nfilters_embeded = int(kwargs.get("nfilters_embeded", 1))
        nlayers_embeded = int(kwargs.get("nlayers_embeded", 1))

        input_model = Sequential()
        for i in range(nlayers_embeded):
            if i == 0:
                nfilters = self.input_dims
                input_model.add(ConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
            elif i == nlayers_embeded-1:
                nfilters = nfilters_embeded
                input_model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded_last, 3, 1, 1, 3, 1, 1))
            else:
                nfilters = nfilters_embeded
                input_model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
        input_model.add(BinaryLinearBNSoftmax(None, self.output_dims))

        input_model.build()
        return input_model

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
        ent_T = kwargs.get("ent_T", None)
        lr = kwargs.get("lr", 0.001)

        chain = Chain(branchweight=branchweight, ent_T=ent_T)
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
            #if k=='nepochs':
                continue
            name = "{}_{}_{}".format(name, k, float(v))
        return name

    def train_model(self, trainset, testset, **kwargs):
        chain, model = self.setup_chain_model(**kwargs)

        nepochs = int(kwargs.get("nepochs", 2))
        ent_T = kwargs.get("ent_T", None)
        name = self.get_name(**kwargs)

        reports = [
         'epoch',
         'main/loss',
         'main/accuracy',
         'validation/main/accuracy',
         'validation/main/memory'
        ]
        trainer = Trainer('{}/{}'.format(self.folder,name), chain, trainset,
                          testset, batchsize=self.batchsize, nepoch=nepochs, resume=True, reports=reports)
        trainer.run()
        
        return trainer, model, chain
