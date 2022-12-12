+++
title = "NN: From Zero to Hero, Lecture 1"
date = 2022-12-04T00:00:00+00:00
lastmod = 2022-12-12T13:21:14+00:00
draft = false
+++

Opinionated translation of the first lecture of
[Andrej Karpathy's course](https://github.com/karpathy/nn-zero-to-hero) into a
step-by-step guide where you're encouraged to come up with solutions on your
own.

<!--more-->


## Goals {#goals}

-   learn what is neural network from first principles
-   learn what is backward propagation / gradient descent
-   build and train a neural network as binary classificator from scratch
-   refine your Python and Jupyter skills
-   learn how PyTorch works under the hood
-   learn how to visualize data as graphs


## Prerequisites {#prerequisites}

-   Basic Python
-   Basic calculus


## Key terms {#key-terms}

-   GraphViz
-   Jupyter
-   PyTorch
-   Python
-   backpropagation
-   binary classification
-   data science
-   data visualization
-   expression graph
-   gradient descent
-   machine learning
-   neural networks


## Further reading {#further-reading}

-   [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)


## Steps {#steps}


### Install Python {#install-python}

-   install Miniconda3: <https://docs.conda.io/en/latest/miniconda.html>
-   create a project directory: `mkdir my-project && cd my-project`
-   create local python environment: `conda create --prefix ./envs`
-   activate it: `conda activate ./envs`


### Create Jupyter notebook {#create-jupyter-notebook}

-   install Jupyter: `pip install jupyter`
-   run Jupyter server: `jupyter notebook`


### Create Value abstraction {#create-value-abstraction}

-   it represents a float value
-   it supports addition and multiplication with other Values
    ```python
      x = Value(1.0)
      y = Value(2.0)
      z = Value(3.0)
      (x + y) * z # Value(9.0)
    ```
-   non-leaf values store its operation and arguments
    ```python
      x = Value(1.0)
      y = Value(2.0)
      z = x + y # Value(3.0, op=<addition>, args=<x and y>)
    ```


### Visualize the resulting expression graph {#visualize-the-resulting-expression-graph}

-   install GraphViz: `pip install graphviz`
-   import relevant class: `from graphviz import Digraph`
-   draw the expression graph:
    -   create graph object
        ```python
            dot = Digraph()
        ```
    -   add Value nodes to graph
        ```python
            dot.node(
                name, # unique node identifier
                label, # contents of the node, format depends on the shape
                shape, # visual shape of the nod
            )
            # dot.node(name='a', label=f'{{ a | {a:.4f} }}', shape='record')
        ```
    -   connect argument nodes to the output nodes
        ```python
            dot.edge(
                name_from, # source node name
                name_to, # destination node name
            )
            # dot.edge('a', 'b')
        ```
    -   given expression `(x + y) * z`

        -   your graph can look like this

        {{< figure src="/ox-hugo/l1-simple-graph.svg" >}}

        -   or like this

        {{< figure src="/ox-hugo/l1alt-simple-graph.svg" >}}


### Implement gradient calculation {#implement-gradient-calculation}

-   gradient is a partial derivative of the final expression
    with respect to the current expression
    ```python
      x = Value(1.0)
      y = Value(2.0)
      z = Value(3.0)
      u = x + y
      v = u * z
    ```

\begin{align}
\text{grad}(v) = \frac{dv}{dv} &= 1\\\\\[5pt]
\text{grad}(u) = \frac{dv}{du} &= \frac{d(u \cdot z)}{du}=z\\\\\[5pt]
\text{grad}(x) = \frac{dv}{dx} &= \frac{dv}{du} \cdot \frac{du}{dx} = z \cdot \frac{d(x + y)}{dx} = z \cdot 1 = z
\end{align}

-   implement `backward()` method which computes the gradients of the whole graph
    when called on the root node
-   hints:
    -   when creating a new Value as a result of some operation, define
        `self._backward` lambda which updates gradients of the argument nodes
    -   consider a case when some Value is used twice


### Implement more operations {#implement-more-operations}

-   subtraction: `x - y = x + (y * -1)`
-   power: `x**k` where `k` is a constant (not a Value)
-   division: `x/y = x * (y**-1)`
-   exp: `x.exp()`
-   tanh: `x.tanh()`


### Create and test expression graph for a single neuron {#create-and-test-expression-graph-for-a-single-neuron}

```python
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8814, label='b')
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()
# ==== expected gradients ====
# x1.grad = -1.5
# w1.grad = 1.0
# x2.grad = 0.5
# w2.grad = 0.0
```


### Create Neuron abstraction {#create-neuron-abstraction}

-   it is defined by a list of weights + bias
-   it is callable with a list of input values, producing a squashed output:
    \\[
        neuron([x\_1, \ldots, x\_n]) = tanh(\sum{w\_i x\_i} + b)
        \\]


### Create Layer abstraction {#create-layer-abstraction}

-   it is defined by a list of neurons
-   it is callable with a list of inputs, producing a list of neuron outputs:
    \\[
        layer([x\_1, \ldots, x\_n]) = [n\_j([x\_1, \ldots, x\_n]) \\,|\\, n\_j \in layer]
        \\]


### Create MLP (Multi-Layer Perceptron) abstraction {#create-mlp--multi-layer-perceptron--abstraction}

-   it is defined by a list of layers
-   it is callable with a list of inputs, producing a list of outputs of the last
    layer:
    \\[
        mlp([x\_1, \ldots, x\_n]) = mlp'(l\_1([x\_1, \ldots, x\_n])) = \ldots = [y\_1,
        \ldots, y\_m]
        \\]
-   for convenience, if the last layer consists only of one neuron return it
    instead of a list


### Create a test dataset for binary classification {#create-a-test-dataset-for-binary-classification}

-   define some sample data, e.g.
    ```python
      # sets of inputs
      xs = [
          [2.0,3.0,-1.0],
          [3.0,-1.0,0.5],
          [0.5,1.0,1.0],
          [1.0,1.0,-1.0],
      ]
      # ground truth (aka expected) outputs
      ys_gt = [1.0,-1.0,-1.0,1.0]
    ```
-   run your MLP on it
    ```python
      # predicted (aka actual) outputs
      ys_pred = [mlp(x) for x in xs]
      # [Value(-0.79),Value(-0.29),Value(0.65),Value(0.23)]
    ```


### Compute the loss {#compute-the-loss}

-   it indicates how good is the MLP prediction
-   there are different loss functions, but we will use Mean Squared Error (MSE)

\\[
loss = \sum\_j(y\_{pred}^j - y\_{gt}^j)^2
\\]


### Update MLP parameters {#update-mlp-parameters}

-   add `parameters()` method to MLP which returns the list of all weights and biases
-   compute the gradients starting from the loss
-   update parameters to decrease the loss
    -   hint: nudge in the opposite direction to the gradient
        ```python
              rate = 0.001
              for p in mlp.parameters():
                  p.data += rate * -p.grad
        ```
-   compute the loss once again and see it getting smaller, which means
    predictions are getting closer to the ground truth


### Create a cycle: Prediction-Loss-Backprop-Update {#create-a-cycle-prediction-loss-backprop-update}

-   iterate N times:
    -   compute the predictions
    -   compute the loss
    -   backprop gradients from the loss
    -   update MLP parameters
-   look at predicted values to see how close they got to the ground truth


### Conclusion {#conclusion}

-   you've just created, trained and used a real neural network!
