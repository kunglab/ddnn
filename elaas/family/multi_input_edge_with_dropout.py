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

class MultiInputEdgeDropoutFamily:
    def __init__(self, folder="_models/multi_input_edge", prefix=None, input_dims=3, output_dims=3,
                 batchsize=10, ninputs=2, merge_function="max_pool_concat", resume=True,
                 drop_comm_train=0, drop_comm_test=0):
        self.ninputs = ninputs
        self.folder = folder
        self.prefix = prefix
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batchsize = batchsize
        self.merge_function = merge_function
        self.resume = resume
        self.drop_comm_train = drop_comm_train
        self.drop_comm_test = drop_comm_test

    def get_configurable_params(self):
        return ["nfilters_embeded", "nlayers_embeded", "nfilters_embeded_last", "nfilters_cloud", "nlayers_cloud", "branchweight", "lr", "ent_T"]

    def generate_model(self, **kwargs):
        nfilters_embeded_last = int(kwargs.get("nfilters_embeded_last", 1))
        nfilters_embeded = int(kwargs.get("nfilters_embeded", 1))
        nlayers_embeded = int(kwargs.get("nlayers_embeded", 1))
        nfilters_cloud = int(kwargs.get("nfilters_cloud", 1))
        nlayers_cloud = int(kwargs.get("nlayers_cloud", 1))
        nfilters_edge = int(kwargs.get("nfilters_edge", 1))
        nlayers_edge = int(kwargs.get("nlayers_edge", 1))

        input_model = Sequential()
        if nlayers_embeded == 1:
            nfilters_embeded_last = nfilters_embeded
        for i in range(nlayers_embeded):
            if i == 0:
                nfilters = self.input_dims
                input_model.add(ConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 2, 1))
            elif i == nlayers_embeded-1:
                nfilters = nfilters_embeded
                input_model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded_last, 3, 1, 1, 3, 2, 1))
            else:
                nfilters = nfilters_embeded
                input_model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 2, 1))
        branch = Sequential()
        branch.add(BinaryLinearBNSoftmax(None, self.output_dims))
        input_model.add(branch)

        model = MultiInputSequential(self.ninputs, merge_function=self.merge_function)
        for i in range(self.ninputs):
            model.add_input(input_model)

        # Local branch
        local_branch = Sequential()
        #local_branch.add(Linear(None, self.output_dims))
        #local_branch.add(BatchNormalization(self.output_dims))
        if self.merge_function in ['concat', 'concat_avg_pool', 'concat_max_pool']:
            local_branch.add(BinaryLinearBNSoftmax(None, self.output_dims))
        model.add_local(local_branch)

        # Edge branches
        #for i in range(nlayers_edge):
        #    if i == 0:
        #        if 'concat' in self.merge_function:
        #            nfilters = self.ninputs*nfilters_embeded_last
        #        else:
        #            nfilters = nfilters_embeded_last
        #    else:
        #        nfilters = nfilters_edge
        #    model.add(BinaryConvPoolBNBST(nfilters, nfilters_edge, 3, 1, 1, 3, 1, 1))
        #
        #    #model.add(Convolution2D(nfilters, nfilters_edge, 3, 1, 1))
        #    #model.add(Activation('relu'))
        #    #model.add(max_pooling_2d(3,1,1))
        #    #model.add(BatchNormalization(nfilters_edge))
        #    #model.add(Activation('relu'))
        #    # Note: should we move pool to before batch norm like in binary?
        #
        #edge_branch = Sequential()
        ##edge_branch.add(Linear(None, self.output_dims))
        ##edge_branch.add(BatchNormalization(self.output_dims))
        #edge_branch.add(BinaryLinearBNSoftmax(None, self.output_dims))
        #model.add(edge_branch)

        # Cloud branches
        for i in range(nlayers_cloud):
            if i == 0:
                if self.merge_function in ['concat', 'avg_pool_concat', 'max_pool_concat']:
                    nfilters = self.ninputs*nfilters_embeded_last
                else:
                    nfilters = nfilters_embeded_last
                if self.drop_comm_train>0:
                    model.add(dropout_comm_train(self.drop_comm_train))
                if self.drop_comm_test>0:
                    model.add(dropout_comm_test(self.drop_comm_test))
            else:
                nfilters = nfilters_cloud
            model.add(BinaryConvPoolBNBST(nfilters, nfilters_cloud, 3, 1, 1, 3, 1, 1))

            #model.add(Convolution2D(nfilters, nfilters_cloud, 3, 1, 1))
            #model.add(Activation('relu'))
            #model.add(max_pooling_2d(3,1,1))
            #model.add(BatchNormalization(nfilters_cloud))
            #model.add(Activation('relu'))
            # Note: should we move pool to before batch norm like in binary?
        #model.add(Linear(None, self.output_dims))
        #model.add(BatchNormalization(self.output_dims))
        model.add(BinaryLinearBNSoftmax(None, self.output_dims))

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
        chain.setup_optimizers('smorms3', lr)
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
         'validation/main/branch0accuracy',
         'validation/main/branch1accuracy',
         'validation/main/branch2accuracy',
         'validation/main/branch3accuracy',
         'validation/main/branch4accuracy',
         'validation/main/branch5accuracy',
         'validation/main/branch6accuracy',
         'validation/main/branch7accuracy',
         'validation/main/branch8accuracy',
         'validation/main/accuracy',
         'validation/main/communication0',
         'validation/main/communication1',
         'validation/main/memory',
         'validation/main/ent_T',
         'validation/main/branch0exit',
         'validation/main/branch1exit',
         'validation/main/branch2exit'
        ]
        trainer = Trainer('{}/{}'.format(self.folder,name), chain, trainset,
                          testset, batchsize=self.batchsize, nepoch=nepochs, resume=self.resume, reports=reports)
        trainer.run()

        return trainer, model, chain
