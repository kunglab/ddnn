# Quick Start
    
    from elaas.elaas import Collection
    import chainer

    mnist = Collection("MNIST")
    train, test = chainer.datasets.get_mnist(ndim=3)

    mnist.add_trainset(train)
    mnist.add_testset(test)
    
    # switch model family
    # mnist.set_model_family() 
    mnist.set_searchspace(
        nfilters_embeded=[1,2],
        nlayers_embeded=[1,2],
        nfilters_cloud=[1,2],
        nlayers_cloud=[1,2],
        lr=[0.001],
        branchweight=[.1]
    )
    def constraintfn(**kwargs):
        # Return true if you want to consider this point, otherwise false
        return True
    mnist.set_constraints(constraintfn)
    # switch chooser
    # mnist.set_chooser()
    
    traces = mnist.train()
    visualize(traces)
    
    # generate c
    # mnist.generate_c((1,28,28))
    
    # generate container
    # mnist.generate_container()