+++
title = "NN: From Zero to Hero, Lecture 1"
lastmod = 2022-12-04T17:49:02+00:00
draft = false
+++

Attempt to translate the first lecture of
[Andrej Karpathy's course](https://github.com/karpathy/nn-zero-to-hero) into a
step-by-step guide where you're encouraged to come up with solutions on your
own.

<!--more-->


## Create Jupyter notebook {#create-jupyter-notebook}

-   install Jupyter: `pip install jupyter`
-   run Jupyter server: `jupyter notebook`


## Create Value abstraction {#create-value-abstraction}

-   it should represent a float value
-   it should support addition and multiplication with other Values
    ```python
      x = Value(1.0)
      y = Value(2.0)
      z = Value(3.0)
      (x + y) * z # Value(9.0)
    ```
-   non-leaf values should store its operation and arguments
    ```python
      x = Value(1.0)
      y = Value(2.0)
      z = x + y # Value(3.0, op=<addition>, args=<x and y>)
    ```


## Visualize the resulting expression graph via GraphViz {#visualize-the-resulting-expression-graph-via-graphviz}

-   install: `pip install graphviz`
-   import: `from graphviz import Digraph`
-   draw the expression graph using GraphViz API:
    -   create graph object
        ```python
            dot = Digraph()
        ```
    -   add graph node
        ```python
            dot.node(
                name, # unique node identifier
                label, # contents of the node, format depends on the shape
                shape, # visual shape of the nod
            )
            # dot.node(name='a', label=f'{{ a | {a:.4f} }}', shape='record')
        ```
    -   add graph edge
        ```python
            dot.edge(
                name_from, # source node name
                name_to, # destination node name
            )
            # dot.edge('a', 'b')
        ```
-


## Implement gradient calculation {#implement-gradient-calculation}

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


## Implement more operations {#implement-more-operations}

-   subtraction: `x - y = x + (y * -1)`
-   power: `x**k` where `k` is a constant (not a Value)
-   division: `x/y = x * (y**-1)`
-   exp: `x.exp()`
-   tanh: `x.tanh()`


## Construct and test expression graph for a single neuron {#construct-and-test-expression-graph-for-a-single-neuron}

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


## Create Neuron abstraction {#create-neuron-abstraction}


## Create Layer abstraction {#create-layer-abstraction}


## Create MLP (Multi-Layer Perceptron) abstraction {#create-mlp--multi-layer-perceptron--abstraction}
