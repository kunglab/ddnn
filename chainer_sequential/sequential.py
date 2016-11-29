import copy, json, types
import chainer
import link
import function
import binary_link
import binary_function
import numpy as np
from chainer import cuda
import inspect
from chainer_ext.functions import entropy
import chainer.functions as F
from chainer import Variable
                
class Sequential(object):
    def __init__(self, stages=[0], weight_initializer="Normal", weight_init_std=1):
        self._layers = []
        self._stages = stages
        self.links = []

        self.weight_initializer = weight_initializer    # Normal / GlorotNormal / HeNormal
        self.weight_init_std = weight_init_std
        self.current_stage = 0

    def add(self, layer):
        if isinstance(layer, Sequential):
            self._layers.append(layer)
        elif isinstance(layer, link.Link) or isinstance(layer, function.Function):
            self._layers.append(layer)
        elif isinstance(layer, function.Activation):
            self._layers.append(layer.to_function())
        elif isinstance(layer, binary_link.BinaryLink) or isinstance(layer, binary_function.BinaryFunction):
            self._layers.append(layer)
        elif isinstance(layer, binary_function.BinaryActivation):
            self._layers.append(layer.to_function())
        else:
            raise Exception()

    def layer_from_dict(self, dict):
        if "_link" in dict:
            if hasattr(link, dict["_link"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(link, dict["_link"])(**args)
            elif hasattr(binary_link, dict["_link"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(binary_link, dict["_link"])(**args)
        if "_function" in dict:
            if hasattr(function, dict["_function"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(function, dict["_function"])(**args)
            elif hasattr(binary_function, dict["_function"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(binary_function, dict["_function"])(**args)
        raise Exception()

    def dict_to_layer_init_args(self, dict):
        args = copy.deepcopy(dict)
        remove_keys = []
        for key, value in args.iteritems():
            if key[0] == "_":
                remove_keys.append(key)
        for key in remove_keys:
            del args[key]
        return args

    def get_weight_initializer(self):
        if self.weight_initializer.lower() == "normal":
            return chainer.initializers.Normal(self.weight_init_std)
        if self.weight_initializer.lower() == "glorotnormal":
            return chainer.initializers.GlorotNormal(self.weight_init_std)
        if self.weight_initializer.lower() == "henormal":
            return chainer.initializers.HeNormal(self.weight_init_std)
        raise Exception()

    def layer_to_chainer_link(self, layer):
        if hasattr(layer, "_link"):
            if layer.has_multiple_weights() == True:
                if isinstance(layer, link.GRU):
                    layer._init = self.get_weight_initializer()
                    layer._inner_init = self.get_weight_initializer()
                elif isinstance(layer, link.LSTM):
                    layer._lateral_init  = self.get_weight_initializer()
                    layer._upward_init  = self.get_weight_initializer()
                    layer._bias_init = self.get_weight_initializer()
                    layer._forget_bias_init = self.get_weight_initializer()
                elif isinstance(layer, link.StatelessLSTM):
                    layer._lateral_init  = self.get_weight_initializer()
                    layer._upward_init  = self.get_weight_initializer()
                elif isinstance(layer, link.StatefulGRU):
                    layer._init = self.get_weight_initializer()
                    layer._inner_init = self.get_weight_initializer()
            else:
                layer._initialW = self.get_weight_initializer()
            return layer.to_link()
        if hasattr(layer, "_function"):
            return layer
        raise Exception()

    def build(self):
        json = self.to_json()
        self.from_json(json)

    def to_dict(self):
        layers = []
        for layer in self._layers:
            config = layer.to_dict()
            if config.get("layers") is not None:
                layers.append(config)
                continue
            dic = {}
            for key, value in config.iteritems():
                if isinstance(value, (int, float, str, bool, type(None), tuple, list, dict)):
                    dic[key] = value
            layers.append(dic)
        return {
            "layers": layers,
            "stages": self._stages,
            "weight_initializer": self.weight_initializer,
            "weight_init_std": self.weight_init_std
        }

    def to_json(self):
        result = self.to_dict()
        return json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

    def from_json(self, str):
        self.links = []
        self._layers = []
        self._stages = []
        attributes = {}
        dict_array = json.loads(str)
        self.from_dict(dict_array)

    def from_dict(self, dict):
        self.weight_initializer = dict["weight_initializer"]
        self.weight_init_std = dict["weight_init_std"]
        self._stages = dict["stages"]
        for i, layer_dict in enumerate(dict["layers"]):
            if layer_dict.get('layers') is not None:
                layer = Sequential(stages=layer_dict["stages"])
                layer.from_dict(layer_dict)
                self.links.append(layer)
                self._layers.append(layer)
            else:
                layer = self.layer_from_dict(layer_dict)
                link = self.layer_to_chainer_link(layer)
                self.links.append(link)
                self._layers.append(layer)


    #def __call__(self, x, test=False):
    #    return self.links(x, test)

    def entropy_filter(self, x, b, ent_T):
        xp = cuda.get_array_module(b)
        eb = entropy(F.softmax(b))

        if xp != np:
            with cuda.get_device(eb.data):
                exited = eb.data < ent_T
            exited = exited.get()
        else:
            exited = eb.data < ent_T

        y_exit = []
        y_cont = []
        for i,idx in enumerate(exited):
            if idx:
                y_exit.append(b[i:i+1])
            else:
                y_cont.append(x[i:i+1])

        if len(y_exit) > 0:
            y_exit = F.vstack(y_exit)
        if len(y_cont) > 0:
            y_cont = F.vstack(y_cont)
        return y_exit,y_cont,exited

    def predict(self, x, ent_T=None, ent_Ts=None, test=True):
        # Return last layer result
        if ent_T is None and ent_Ts is None:
            return self(x, test)
        elif ent_Ts is None:
            ent_Ts = [ent_T]

        num = x.shape[0]
        bs = []
        exit_i = 0
        for i, link in enumerate(self.links):
            if isinstance(link, Sequential):
                b = link(x, test=test)
                b = b[0]
                y_exit,y_cont,exited = self.entropy_filter(x, b, ent_Ts[min(exit_i,len(ent_Ts)-1)])
                exit_i = exit_i + 1
                b = y_exit
                bs.append((b,exited))
                x = y_cont
                if len(x) == 0:
                    break
            elif isinstance(link, function.dropout):
                x = link(x, train=not test)
            elif isinstance(link, chainer.links.BatchNormalization):
                x = link(x, test=test)
            elif hasattr(link,'__call__') and 'train' in inspect.getargspec(link.__call__)[0]:
                x = link(x, train=not test)
            elif hasattr(link,'__call__') and 'test' in inspect.getargspec(link.__call__)[0]:
                x = link(x, test=test)
            else:
                x = link(x)
        
        if len(x) > 0:
            bs.append((x,[True]*x.shape[0]))
        ys = [None]*num
        for b,exited in bs:
            i = 0
            j = 0
            for exit in exited:
                while ys[i] is not None:
                    i = i + 1
                if exit:
                    ys[i] = b[j]
                    j = j + 1
                i = i + 1
        return F.vstack(ys), exited
    
    def set_current_stage(self, stage):
        self.current_stage = stage
        for i, link in enumerate(self.links):
            if isinstance(link, Sequential):
                link.set_current_stage(stage)
                
    def get_current_stage(self):
        return self.current_stage
        
    def __call__(self, x, test=False):
        bs = []
        numlinks = len(self.links)
        for i, link in enumerate(self.links):
            if isinstance(link, Sequential):
                # detach if in different stages
                #if reduce(lambda x,y: x and y, [stage not in link._stages for stage in self._stages]):
                if self.current_stage not in link._stages:
                    y = Variable(x.data, x.volatile)
                else:
                    y = x
                b = link(y, test=test)
                bs.append(b[0])
                # Currently not support branch inside a branch
            elif isinstance(link, function.dropout):
                x = link(x, train=not test)
            elif isinstance(link, chainer.links.BatchNormalization):
                x = link(x, test=test)
            elif hasattr(link,'__call__') and 'train' in inspect.getargspec(link.__call__)[0]:
                #print("train",link)
                x = link(x, train=not test)
            elif hasattr(link,'__call__') and 'test' in inspect.getargspec(link.__call__)[0]:
                #print("test",link)
                x = link(x, test=test)
            else:
                x = link(x)
            # do not update this branch if not the current stage
            if self.current_stage not in self._stages:
                x.unchain_backward()
        bs.append(x)
        #print(bs)
        return tuple(bs)

    def generate_call(self):
        link_idx = 0
        text = ""

        text += "void compute(){\n"

        l = self.links[0]
        text += "  {name}(input, temp1);\n".format(name=l.cname + str(link_idx))

        link_idx += 1

        lastlink = self.links[-1]

        for l in self.links[1:-1]:
            if isinstance(l, Sequential): # branch off to the first branch
                lastlink = l.links[-1]
                for l in l.links[:-1]:
                    if link_idx % 2 == 1:
                        text += "  {name}(temp1, temp2);\n".format(name=l.cname + str(link_idx))
                    else:
                        text += "  {name}(temp2, temp1);\n".format(name=l.cname + str(link_idx))
                    link_idx = link_idx + 1
                break
            if link_idx % 2 == 1:
                text += "  {name}(temp1, temp2);\n".format(name=l.cname + str(link_idx))
            else:
                text += "  {name}(temp2, temp1);\n".format(name=l.cname + str(link_idx))
            link_idx = link_idx + 1

        l = lastlink

        if link_idx % 2 == 1:
            text += "  {name}(temp1, output);\n".format(name=l.cname + str(link_idx))
        else:
            text += "  {name}(temp2, output);\n".format(name=l.cname + str(link_idx))
        text += "}"

        return text

    def generate_c(self, shape, name="main"):
        text = """
#include "util.h"
"""
        h = np.random.random([1]+list(shape)).astype(np.float32)
        #inp = ",".join([str(item) for item in h.flatten().tolist()])

        input_size = h.size
        inter_sizes = []
        for i, link in enumerate(self.links):
            if isinstance(link, Sequential): # branch off to the first branch only for now
                for j, link in enumerate(link.links):
                    if hasattr(link, 'generate_c'):
                        inter_sizes.append(link.temp_mem(h.shape))
                        text += link.generate_c(i+j, h.shape)
                    h = link(h)
                break
            if hasattr(link, 'generate_c'):
                inter_sizes.append(link.temp_mem(h.shape))
                text += link.generate_c(i, h.shape)
            h = link(h)
        inter_size = int(np.max(inter_sizes))

        text += """
char name[] = "{name}";
float input[{input_size}] = {{{inp}}};
uint8_t output[1] = {{0}};
uint8_t temp1[{inter_size}] = {{0}}; // Allocate large enough
uint8_t temp2[{inter_size}] = {{0}}; // Allocate large enough
""".format(name=name,input_size=input_size,inter_size=inter_size,inp="0")

        text += self.generate_call()
        return text
