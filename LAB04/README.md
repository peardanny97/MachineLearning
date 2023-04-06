# LAB 04 - Part 1: Hidden Markov Model (HMM)

Copyright (C) Data Science & AI Laboratory, Seoul National University. This material is for educational uses only. Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. Written by Nohil Park, May 2021.

Now, you're going to leave behind your implementations and instead migrate to one of popular deep learning frameworks, **PyTorch**. <br>
In this notebook, you will learn how to train a Hidden Markov Model (HMM) for sequential modeling. <br>
You need to follow the instructions to **complete 5 TODO sections and explain them if needed.**

You will see:
- what HMM is and when you might want to use it;
- the so-called "three problems" of an HMM; and 
- how to implement HMM in PyTorch.

### Some helpful tutorials and references for assignment #4:
- [1] Pytorch official documentation. [[link]](https://pytorch.org/docs/stable/index.html)
- [2] Stanford CS231n lectures. [[link]](http://cs231n.stanford.edu/)
- [3] https://ratsgo.github.io/machine%20learning/2017/03/18/HMMs/
- [4] https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/

## The HMM setup

An HMM models a system that is in a particular state at any given time and produces an output that depends on that state. 

At each timestep or clock tick, the system randomly decides on a new state and jumps into that state. The system then randomly generates an observation. The states are "hidden": we can't observe them. (In the cities/selfies analogy, the unknown cities would be the hidden states, and the selfies would be the observations.)

Let's denote the sequence of states as $\mathbf{z} = \{z_1, z_2, \dots, z_T \}$, where each state is one of a finite set of $N$ states, and the sequence of observations as $\mathbf{x} = \{x_1, x_2, \dots, x_T\}$. The observations could be discrete, like letters, or real-valued, like audio frames.

An HMM makes two key assumptions:
- **Assumption 1:** The state at time $t$ depends *only* on the state at the previous time $t-1$. 
- **Assumption 2:** The output at time $t$ depends *only* on the state at time $t$.

These two assumptions make it possible to efficiently compute certain quantities that we may be interested in.

## Components of an HMM
An HMM has three sets of trainable parameters.
  
- The **transition model** is a square matrix $A$, where $A_{s, s'}$ represents $p(z_t = s|z_{t-1} = s')$, the probability of jumping from state $s'$ to state $s$. 

- The **emission model** $b_s(x_t)$ tells us $p(x_t|z_t = s)$, the probability of generating $x_t$ when the system is in state $s$. For discrete observations, which we will use in this notebook, the emission model is just a lookup table, with one row for each state, and one column for each observation. For real-valued observations, it is common to use a Gaussian mixture model or neural network to implement the emission model. 

- The **state priors** tell us $p(z_1 = s)$, the probability of starting in state $s$. We use $\pi$ to denote the vector of state priors, so $\pi_s$ is the state prior for state $s$.

Let's program an HMM class in PyTorch.

Let's try hard-coding an HMM for generating fake words. (We'll also add some helper functions for encoding and decoding strings.)

We will assume that the system has one state for generating vowels and one state for generating consonants, and the transition matrix has 0s on the diagonal---in other words, the system cannot stay in the vowel state or the consonant state for more than one timestep; it has to switch.

Since we pass the transition matrix through a softmax, to get 0s we set the unnormalized parameter values to $-\infty$.

**TODO 1: Try sampling from our hard-coded model.**


Sampled outputs should have the form of: <br>
```
x: adarituhij
z: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
```

## Problem 2. The Three Problems

The "three problems" that need to be solved before you can effectively use an HMM are:
- Problem 1: How do we efficiently compute $p(\mathbf{x})$?
- Problem 2: How do we find the most likely state sequence $\mathbf{z}$ that could have generated the data? 
- Problem 3: How do we train the model?

In the rest of the notebook, we will see how to solve each problem and implement the solutions in PyTorch.

### Problem 2-1. How do we compute $p(\mathbf{x})$?


#### *Why?*
Why might we care about computing $p(\mathbf{x})$? Here's two reasons.
* Given two HMMs, $\theta_1$ and $\theta_2$, we can compute the likelihood of some data $\mathbf{x}$ under each model, $p_{\theta_1}(\mathbf{x})$ and $p_{\theta_2}(\mathbf{x})$, to decide which model is a better fit to the data. 

  (For example, given an HMM for English speech and an HMM for French speech, we could compute the likelihood given each model, and pick the model with the higher likelihood to infer whether the person is speaking English or French.)
* Being able to compute $p(\mathbf{x})$ gives us a way to train the model, as we will see later.

#### *How?*
Given that we want $p(\mathbf{x})$, how do we compute it?

We've assumed that the data is generated by visiting some sequence of states $\mathbf{z}$ and picking an output $x_t$ for each $z_t$ from the emission distribution $p(x_t|z_t)$. So if we knew $\mathbf{z}$, then the probability of $\mathbf{x}$ could be computed as follows:

$$p(\mathbf{x}|\mathbf{z}) = \prod_{t} p(x_t|z_t) p(z_t|z_{t-1})$$

However, we don't know $\mathbf{z}$; it's hidden. But we do know the probability of any given $\mathbf{z}$, independent of what we observe. So we could get the probability of $\mathbf{x}$ by summing over the different possibilities for $\mathbf{z}$, like this:

$$p(\mathbf{x}) = \sum_{\mathbf{z}} p(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) = \sum_{\mathbf{z}} p(\mathbf{z}) \prod_{t} p(x_t|z_t) p(z_t|z_{t-1})$$

The problem is: if you try to take that sum directly, you will need to compute $N^T$ terms. This is impossible to do for anything but very short sequences. For example, let's say the sequence is of length $T=100$ and there are $N=2$ possible states. Then we would need to check $N^T = 2^{100} \approx 10^{30}$ different possible state sequences.

We need a way to compute $p(\mathbf{x})$ that doesn't require us to explicitly calculate all $N^T$ terms. For this, we use the forward algorithm.

________

<u><b>The Forward Algorithm</b></u>

> for $s=1 \rightarrow N$:\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\alpha_{s,1} := b_s(x_1) \cdot \pi_s$ 
> 
> for $t = 2 \rightarrow T$:\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for $s = 1 \rightarrow N$:\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
> $\alpha_{s,t} := b_s(x_t) \cdot \underset{s'}{\sum} A_{s, s'} \cdot \alpha_{s',t-1} $
> 
> $p(\mathbf{x}) := \underset{s}{\sum} \alpha_{s,T}$\
> return $p(\mathbf{x})$
________

The forward algorithm is much faster than enumerating all $N^T$ possible state sequences: it requires only $O(N^2T)$ operations to run, since each step is mostly multiplying the vector of forward variables by the transition matrix. (And very often we can reduce that complexity even further, if the transition matrix is sparse.)

There is one practical problem with the forward algorithm as presented above: it is prone to underflow due to multiplying a long chain of small numbers, since probabilities are always between 0 and 1. Instead, let's do everything in the log domain. In the log domain, a multiplication becomes a sum, and a sum becomes a ${\text{logsumexp}}$.

Let $x \in \mathbb{R}^n$, ${\text{logsumexp}} (x)$ is defined as:

$${\text{logsumexp}} (x) = \log (\sum_{i} \exp (x_i)).$$

Numerically, ${\text{logsumexp}}$ is similar to ${\max}$: in fact, it’s sometimes called the “smooth maximum” function. For example:

$$\max ([1, 2, 3]) = 3 $$
$${\text{logsumexp}} ([1, 2, 3]) = 3.4076$$

________

<u><b>The Forward Algorithm (Log Domain)</b></u>

> for $s=1 \rightarrow N$:\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\text{log }\alpha_{s,1} := \text{log }b_s(x_1) + \text{log }\pi_s$ 
> 
> for $t = 2 \rightarrow T$:\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for $s = 1 \rightarrow N$:\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
> $\text{log }\alpha_{s,t} := \text{log }b_s(x_t) +  \underset{s'}{\text{logsumexp}} \left( \text{log }A_{s, s'} + \text{log }\alpha_{s',t-1} \right)$
> 
> $\text{log }p(\mathbf{x}) := \underset{s}{\text{logsumexp}} \left( \text{log }\alpha_{s,T} \right)$\
> return $\text{log }p(\mathbf{x})$
________

**TODO 2**: Now that we have a numerically stable version of the forward algorithm, let's implement it in PyTorch. 

When using the vowel <-> consonant HMM from above, notice that the forward algorithm returns $-\infty$ for $\mathbf{x} = ``\text{abb"}$. That's because our transition matrix says the probability of vowel -> vowel and consonant -> consonant is 0, so the probability of $``\text{abb"}$ happening is 0, and thus the log probability is $-\infty$.

The Viterbi algorithm looks somewhat gnarlier than the forward algorithm, but it is essentially the same algorithm, with two tweaks: 1) instead of taking the sum over previous states, we take the max; and 2) we record the argmax of the previous states in a table, and loop back over this table at the end to get $\mathbf{z}^*$, the most likely state sequence. (And like the forward algorithm, we should run the Viterbi algorithm in the log domain for better numerical stability.) 

**TODO 3**: Let's add the Viterbi algorithm to our PyTorch model:

### Problem 2-3. Comparison: Forward vs. Viterbi

**TODO 4**: Compare the "forward score" (the log probability of all possible paths, returned by the forward algorithm) with the "Viterbi score" (the log probability of the maximum likelihood path, returned by the Viterbi algorithm). You can print out model's outputs.

### Problem 2-4. How do we train the model?

Earlier, we hard-coded an HMM to have certain behavior. What we would like to do instead is have the HMM learn to model the data on its own. And while it is possible to use supervised learning with an HMM (by hard-coding the emission model or the transition model) so that the states have a particular interpretation, the really cool thing about HMMs is that they are naturally unsupervised learners, so they can learn to use their different states to represent different patterns in the data, without the programmer needing to indicate what each state means.

**TODO 5**: Try to decode some different sequences (e.g. proper words, misspelled words) with the Viterbi algorithm and give an explanation about the results.

# LAB 04 - Part 2: Recurrent Neural Networks (RNNs)

Copyright (C) Data Science & AI Laboratory, Seoul National University. This material is for educational uses only. Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. Written by Nohil Park, May 2021.

In this notebook, you will learn how to train a Recurrent Neural Network (RNN) for sequential modeling. Specifically, you will build a character-level RNN to generate names from different languages. <br> 
You need to follow the instructions to **complete 4 TODO sections and explain them if needed.**

You will see:
- how to implement a character-level RNN in PyTorch;
- train the network; and
- sampling sequences from the network.

### Some helpful tutorials and references for assignment #4:
- [1] Pytorch official documentation. [[link]](https://pytorch.org/docs/stable/index.html)
- [2] Stanford CS231n lectures. [[link]](http://cs231n.stanford.edu/)
- [3] https://ratsgo.github.io/machine%20learning/2017/03/18/HMMs/
- [4] https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/

Problem 1. Creating the Network
====================

**TODO 1**: Instead of using ``torch.nn.RNN``, build your own RNN network.

RNN architecture in order:
- ``i2h`` : a linear layer for the hiddent
- ``i2o`` : a linear layer for the output
- ``o2o`` : a linear layer after combining hidden and output
- ``dropout`` : randomly zeros parts of its
input <https://arxiv.org/abs/1207.0580>`__ with a given probability
(here 0.1) and is usually used to fuzz inputs to prevent overfitting.
- ``softmax`` : LogSoftmax for the final output

Category tensor is a one-hot vector just like the letter input and it is concatenated with the input and hidden. Output is interpreted as the probability of the next letter. When sampling, the most likely output letter is used as the next input letter.

Training
=========
Preparing for Training
----------------------

First of all, helper functions to get random pairs of (category, line):

For each timestep (that is, for each letter in a training word) the
inputs of the network will be
``(category, current letter, hidden state)`` and the outputs will be
``(next letter, next hidden state)``. So for each training set, we'll
need the category, a set of input letters, and a set of output/target
letters.

Since we are predicting the next letter from the current letter for each
timestep, the letter pairs are groups of consecutive letters from the
line - e.g. for ``"ABCD<EOS>"`` we would create ("A", "B"), ("B", "C"),
("C", "D"), ("D", "EOS").

The category tensor is a `one-hot
tensor`__ of size
``<1 x n_categories>``. When training we feed it to the network at every
timestep - this is a design choice, it could have been included as part
of initial hidden state or some other strategy.


Problem 2. Training the Network
--------------------

In contrast to classification, where only the last output is used, we
are making a prediction at every step, so we are calculating loss at
every step.

The magic of autograd allows you to simply sum these losses at each step
and call backward at the end.

**TODO 2**: Define a loss function, learning rate and hidden size for RNN training.

Problem 3. Sampling from the Network
====================

**TODO 3**: To sample we give the network a letter and ask what the next one is,
feed that in as the next letter, and repeat until the EOS token.

-  Create tensors for input category, starting letter, and empty hidden
   state
-  Create a string ``output_name`` with the starting letter
-  Up to a maximum output length,

   -  Feed the current letter to the network
   -  Get the next letter from highest output, and next hidden state
   -  If the letter is EOS, stop here
   -  If a regular letter, add to ``output_name`` and continue

-  Return the final name

.. Note::
   Rather than having to give it a starting letter, another
   strategy would have been to include a "start of string" token in
   training and have the network choose its own starting letter.

**TODO 4**: Sample 12 different names from 4 languages (i.e. 3 names per language). Use the ``samples`` function defined above.
