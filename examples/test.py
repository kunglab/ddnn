import sys
sys.path.append('..')

import chainer

from elaas.elaas import Collection
from elaas.family.simple import SimpleHybridFamily
from visualize import visualize
import deepopt.chooser
<<<<<<< HEAD

save_dir = '/mnt/data/'

=======
#save_dir = '/mnt/data/'
save_dir = "_models"

mnist = Collection('simple_hybrid', save_dir, nepochs=10)
mnist.set_model_family(SimpleHybridFamily)
>>>>>>> f097932812b3d06c54338b0ab0c87a4098a48b9c
train, test = chainer.datasets.get_mnist(ndim=3)

mnist = Collection('simple_hybrid', nepochs=10)
mnist.set_model_family(SimpleHybridFamily(folder=save_dir+'hybrid/'))
mnist.add_trainset(train)
mnist.add_testset(test)

mnist.set_searchspace(
    nfilters_embeded=[1],
    nlayers_embeded=[1],
    nfilters_cloud=[1],
    nlayers_cloud=[1],
    lr=[0.001],
    branchweight=[.1]
)

def constraintfn(**kwargs):
    #TODO: change to memory cost
    if kwargs['nfilters_embeded'] > 2:
        return False

    return True

mnist.set_constraints(constraintfn)

# switch chooser
mnist.set_chooser(deepopt.chooser.EpochChooser(k=1))

# currently optimize based on the validation accuracy of the main model
traces = mnist.train()
# visualize(traces)


# generate c
# mnist.generate_c((1,28,28))

# generate container
# mnist.generate_container()

# get traces for the collection 
# mnist = Collection('simple_hybrid', save_dir)
# traces = mnist.get_do_traces()