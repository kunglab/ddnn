from chainer_sequential.multiinputchain import MultiInputChain
from deepopt.trainer import Trainer
import chainer
import chainer.serializers as S
from chainer_sequential.sequential import Sequential
from chainer_sequential.multiinputsequential import MultiInputSequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F

class MultiInputFamily:
    def __init__(self, folder="_models/multi_input", prefix=None, input_dims=3, output_dims=3, batchsize=10, ninputs=2, resume=True):
        self.ninputs = ninputs
        self.folder = folder
        self.prefix = prefix
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batchsize = batchsize
        self.resume = resume

    def get_configurable_params(self):
        return ["nfilters_embeded", "nlayers_embeded", "nfilters_cloud", "nlayers_cloud", "branchweight", "lr", "ent_T"]

    def generate_model(self, **kwargs):
        nfilters_embeded = int(kwargs.get("nfilters_embeded", 1))
        nlayers_embeded = int(kwargs.get("nlayers_embeded", 1))
        nfilters_cloud = int(kwargs.get("nfilters_cloud", 1))
        nlayers_cloud = int(kwargs.get("nlayers_cloud", 1))

        input_model = Sequential()
        for i in range(nlayers_embeded):
            if i == 0:
                nfilters = self.input_dims
                input_model.add(ConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
            else:
                nfilters = nfilters_embeded
                input_model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
        branch = Sequential()
        branch.add(BinaryLinearBNSoftmax(None, self.output_dims))
        input_model.add(branch)

        model = MultiInputSequential(self.ninputs)
        for i in range(self.ninputs):
            model.add_input(input_model)
                
        # float branch
        for i in range(nlayers_cloud):
            if i == 0:
                nfilters = self.ninputs*nfilters_embeded
            else:
                nfilters = nfilters_cloud
            model.add(Convolution2D(nfilters, nfilters_cloud, 3, 1, 1))
            model.add(max_pooling_2d(3,1,1))
            model.add(BatchNormalization(nfilters_cloud))
            model.add(Activation('bst'))
            # Note: should we move pool to before batch norm like in binary?
        model.add(Linear(None, self.output_dims))
        model.add(BatchNormalization(self.output_dims))
        
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
        ent_T = kwargs.get("ent_T", None)
        lr = kwargs.get("lr", 0.001)

        chain = MultiInputChain(branchweight=branchweight, ent_T=ent_T)
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
            name = "{}_{}_{}".format(name, k, float(v))
        return name

    def train_model(self, trainset, testset, **kwargs):
        chain, model = self.setup_chain_model(**kwargs)

        nepochs = int(kwargs.get("nepochs", 2))
        name = self.get_name(**kwargs)

        reports = [
         'validation/main/branch0accuracy',
         'validation/main/branch1accuracy',
         'validation/main/branch2accuracy',
         'validation/main/branch3accuracy',
         'validation/main/branch4accuracy',
         'validation/main/branch5accuracy',
         'validation/main/branch6accuracy',
         'validation/main/branch7accuracy'
        ]
        trainer = Trainer('{}/{}'.format(self.folder,name), chain, trainset,
                          testset, batchsize=self.batchsize, nepoch=nepochs, resume=self.resume, reports=reports)
        trainer.run()
        
        return trainer, model
