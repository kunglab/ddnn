ChainerをKerasっぽく書きたいという思いから作りました。

ただしKerasを使ったことがないのであくまでそれっぽいというだけです。

内部的にはChainerのLinkやFunctionをユーザーが定義したとおりの順に並べて実行します。

## Requirements
- Chainer 1.17

## Usage

```
from link import Linear, BatchNormalization
from function import Activation
from chain import Chain

x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

model = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
model.add(Linear(28*28, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(None, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(500, 28*28, use_weightnorm=True))
model.build()

chain = Chain()
chain.add_sequence(model)
chain.setup_optimizers("adam", 0.001, momentum=0.9, weight_decay=0.000001, gradient_clipping=10)

for i in xrange(100):
	y = chain(x)
	loss = F.mean_squared_error(x, y)
	chain.backprop(loss)
	print float(loss.data)

chain.save("model")
```

## Getting chainer.links.Link objects

```
model = Sequential()
...
model.build()

for i, link in enumerate(model.links):
	if isinstance(link, chainer.link.Link):
		...
```

## JSON

```
json_str = model.to_json()
model.from_json(json_str)
```

## Adding activation function

```
model.add(function.Activation("relu"))
```

or

```
model.add(function.relu())
```

## Dropout

```
model.add(function.dropout())
```

## Adding gaussian noise

```
model.add(function.gaussian_noise(std=0.5))
```

## Weight Normalization

```
model.add(link.Linear(500, 500, use_weightnorm=True))
model.add(link.Convolution2D(64, 128, ksize=4, stride=2, pad=0, use_weightnorm=True))
model.add(link.Deconvolution2D(64, 32, ksize=4, stride=2, pad=1, use_weightnorm=True))
```

## Initializing weights

```
model = Sequential(weight_initializer="Normal", weight_init_std=0.05)
model = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
model = Sequential(weight_initializer="HeNormal", weight_init_std=0.05)
```

## DCGAN

```
import util

image_width = 96

disciminator = Sequential()
disciminator.add(Convolution2D(3, 64, ksize=4, stride=2, pad=1))
disciminator.add(Activation("elu"))
disciminator.add(Convolution2D(64, 128, ksize=4, stride=2, pad=1))
disciminator.add(Activation("elu"))
disciminator.add(Convolution2D(128, 256, ksize=4, stride=2, pad=1))
disciminator.add(Activation("elu"))
disciminator.add(Linear(None, 1))
disciminator.add(sigmoid())
disciminator.build()

# compute projection width
input_width = util.get_in_size_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

# compute required paddings
paddings = util.get_paddings_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

generator = Sequential()
generator.add(Linear(100, 64 * input_width ** 2))
generator.add(BatchNormalization(64 * input_width ** 2))
generator.add(Activation("relu"))
generator.add(reshape((-1, 64, input_width, input_width)))
generator.add(Deconvolution2D(64, 32, ksize=4, stride=2, pad=paddings.pop(0)))
generator.add(BatchNormalization(32))
generator.add(Activation("relu"))
generator.add(Deconvolution2D(32, 16, ksize=4, stride=2, pad=paddings.pop(0)))
generator.add(BatchNormalization(16))
generator.add(Activation("relu"))
generator.add(Deconvolution2D(16, 3, ksize=4, stride=2, pad=paddings.pop(0)))
generator.build()
```