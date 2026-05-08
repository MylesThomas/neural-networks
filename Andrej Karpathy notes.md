link to video: https://x.com/neil_xbt/status/2047512855155528054?s=12

What we will do:
- Define a neural network
- Train the neural network

micrograd:
- https://github.com/karpathy/micrograd
- repo he put on github a few years ago
- what is micrograd?
    - autograd engine for neural networks
    - "it implements backpropagation"
    - backpropagation: allows you to efficiently compute the gradient of some kind of loss function, with respect to the weights of the neural network
    - "therefore, we increase the accuracy of the neural network"
    - at mathematical core of any neural network training network ie. JAX, PyTorch, TensorFlow, etc.

example usage:

```python
-- Slightly contrived example, showing a number of possible supported operations
from micrograd.engine import Value

# Overall: building out expression graph
# input: a/b
# output: g

# (micrograd builds out the mathemtical expression graph)

# 2 inputs: a and b
# we wrap those values in Value() objects, which wraps the numbers
# they get transformed
a = Value(-4.0)
b = Value(2.0)

c = a + b # child nodes of c: a / b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass

# "not only can we do a forward pass, where output is g ie. 24.7041, but we can also call .backward()
# "initaliazes backpropagation at node g"
# to compute the gradient of g with respect to a and b
# "will start at g, go backwards, and recursively apply the chain rule from calculus"
# what this lets us do: evaluate the derivate of g, with respect to internal nodes (e/d/c) and input nodes (a/b)
# then: we can query this derivative with respone to a, or b, (138 / 645 below)
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

# why this is important:
# - tells us how a/b are affecting g, through this mathematical expression
# - a.grad being 138: if we slighlty make a larger, g will get larger by 138 (ie. slope of +138)
# - b.grad being 645: if we slighlty make b larger, g will get larger by 645 (ie. slope of +645)

# note: this specific expression is completely meaningless, but it's a simple example to show how backpropagation works (flexes the operations that micrograd supports)
# - neural networks are just like this one, but less crazy actually

```

neural network: mathemtical expression (just a certain class)
- take input data
- take weights of a neural network
- output: predictions of neural net OR loss function

reminder: backpropogation is more general - does not care about neurl networks, just cases about arbitrary mathemtical expressions
- we use this to machinery train neural networks

one more note - micrograd is scalar value autograd engine
- example: -4.0, 2.0
- allows us to not have to deal with n-dimensional tensors in many NN libraries
- training bigger networks: need to use these tensors
    - none of math changes, just dont for efficiency
    - what they do:
        - take scalar values
        - package up into tensors
            - "a race of these scalars"
        - make operations on large arrays
            - takes advantage of parallelism on a CPU (so it runs faster)

this is why he wrote micrograd...
- understand how things work at a mental
- THEN later, you can speed it up
- "my claim: micrograd is all you need to train neural networks, and everything else is efficiency"
    - you may think micrograd is very compolicated code then, but it is not

2 files
- engine.py: actual backpropogation, giving power to neural networks, is 100 lines of code
- nn.py: NN library, built on top of autograd engine
    - 3 things defined:
        - Neuron
        - Layer of Neurons
        - MultiLayer Perceptron (sequence of layers)

"it is just a total joke! lots of power from 150 lines of code"

diving right into implementing micrograd, step by step

