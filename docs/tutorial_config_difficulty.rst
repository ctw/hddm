.. index:: Tutorial
.. _chap_tutorial_config:

****************************************
Getting started: Creating a simple model
****************************************

Imagine that we collected data from one subject on the moving dots or
coherent motion task. In this task, participants indicate via keypress
in which of two directions dots on a screen are moving. However,
depending on the difficulty, some of the dots are moving into random
directions (see the figure). In our working example, we used two
conditions, an easy and a hard condition. In the following, we will
walk through the steps on creating a model in HDDM to estimate the
underlying psychological decision making parameters of this task.

..  figure:: moving_dots.jpg
    :scale: 20%

The easiest way to use HDDM is to create a configuration file. First,
you have to prepare your data to be in a specific format (csv). The
data from our was simulated from the DDM so that we know the true
parameters. The data file can be found in the examples directory and
is named simple_difficulty.csv. Lets take a look at what it looks
like:

.. literalinclude :: ../hddm/examples/simple_difficulty.csv
   :lines: 1,101-105,579-582

The first line contains the header and specifies which columns contain
which data.

IMPORTANT: There must be one column named 'rt' and one named
'response'. 

The rows following the header contain the reaction time of the trial,
followed by a comma, followed by the response made (e.g. 1=correct,
0=error or 1=left, 0=right), followed by the difficulty of the
trail. Note, that 'difficulty' is just an example, you can call them
whatever you like.

The following configuration file specifies a model in which
drift-rate depends on difficulty:

.. literalinclude :: ../hddm/examples/simple_difficulty.conf

The [model] tag specifies that parameters after the tag are model
parameters. In this case, the 'data' parameter tells HDDM where the
input data is to be found.

The optional tag [depends_on] specifies DDM parameters that depend on
data. In our case, we want to estimate separate drift-rates (v) for
the conditions found in the data column 'difficulty'.

The optional [mcmc] tag specifies parameter of the Markov chain
Monte-Carlo estimation such as how many samples to draw from the
posterior and how many samples to discard as burn-in (often, it takes
the MCMC chains some time to converge to the true posterior).

Our model specification is now complete and we can fit the model by
calling hddmfit:

::

    hddmfit simple_difficulty.conf simple_difficulty.csv

The first argument tells HDDM which model specification to use, the
second argument is the data file to apply the model to.

Calling hddmfit in this way will generate the following output (note
that the numbers will be slightly different each time you run this):

::

    Creating model...
    Sampling: 100% [0000000000000000000000000000000000] Iterations: 10000

       name       mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a         :  2.029  0.034  1.953  2.009  2.028  2.049  2.090  0.002
    t         :  0.297  0.007  0.282  0.292  0.297  0.302  0.311  0.001
    v('easy',):  0.992  0.051  0.902  0.953  0.987  1.028  1.102  0.003
    v('hard',):  0.522  0.049  0.429  0.485  0.514  0.561  0.612  0.002

    logp: -1171.276303
    DIC: 2329.069932

The parameters of DDM are usually abbreviated and have the following meaning:

    * a: threshold
    * t: non-decision time
    * v: drift-rate
    * z: bias (optional)
    * V: inter-trial variability in drift-rate (optional)
    * Z: inter-trial variability in bias (optional)
    * T: inter-trial variability in non-decision time (optional)

Because we used simulated data in this example, we know the true
parameters that generated the data (i.e. a=2, t=0.3, v_easy=1,
v_hard=0.5). As you can see, the mean posterior values are very close
to the true parameters -- our estimation worked! However, often we are
not only interested in the best fitting value but also how confident
we are in that estimation and how good other values are fitting. This
is one of advantages of the Bayesian approach -- it gives us the
complete posterior distribution. So the next columns are statistics on
the shape of the distribution, such as the standard deviation and
different quantiles to give you a feel for how certain you can be in
the estimates.

Lastly, logp and DIC give you a measure of how well the model fits the
data. These values are not all that useful if looked at in isolation
but they provide a tool to do model comparison. Logp is the summed
log-likelihood of the best-fitting values (higher is better). DIC
stands for deviance information criterion and is a measure that takes
model complexity into account. Lower values are better.

:Excercise:

    Create a new model that ignores the different difficulties (i.e. only
    estimate one drift-rate). Compare the resulting DIC score with that of
    the previous model -- does the increased complexity of the first model
    result in a sufficient increase in model fit to justify using it? Why
    does the drift-rate estimate of the second model make sense?

Output plots
************

In addition, HDDM generates some useful plots such as the posterior
predictive probability density on top of the normalized RT
distribution for each condition:

.. figure:: ../hddm/examples/plots/simple_difficulty_easy.png
   :scale: 40%

.. figure:: ../hddm/examples/plots/simple_difficulty_hard.png
   :scale: 40%

Note that error responses have been mirrored along the y-axis to
display both RT distributions in one plot. 

These plots allow you to see how good the estimation fits your
data. Here, we also see that our subject makes more errors and are
slower in the difficult condition which is covered well by the
decreased drift-rate parameter.

Moreover, HDDM generates the trace and histogram of the posterior
samples. As pointed out in the introduction, we can rarely compute the
posterior analytically so we have to estimate it. One standard method
is MCMC which allows you to draw samples from the posterior. On the
left side of the plot we see the trace of this sampling. The main
thing to look out for is if the chain drifts around (i.e. has not
converged) or if there are periods where it seems stuck in one place
(i.e. proposal distribution too wide). In our case the chain of the
parameter a seems to have converged nicely to the correct value. This
is also illustrated in the right side plot which is the histogram of
the trace and gives a feel for how to the posterior distribution looks
like. In our case, it looks like a normal distribution centered around
a value close to 2.

.. figure:: ../hddm/examples/plots/simple_difficulty_trace_a.png
   :scale: 40%

Now we are ready for :ref:`part two of the tutorial <chap_tutorial_config_subjects>`.