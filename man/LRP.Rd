% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LRP.R
\name{LRP}
\alias{LRP}
\title{Layer-wise relevance propagation (LRP)}
\description{
This is an implementation of the \emph{layer-wise relevance propagation
(LRP)} algorithm introduced by Bach et al. (2015). It's a local method for
interpreting a single element of the dataset and calculates the relevance
scores for each input feature to the model output. The basic idea of this
method is to decompose the prediction score of the model with respect to
the input features, i.e.,
\deqn{f(x) = \sum_i R(x_i).}
Because of the bias vector that absorbs some relevance, this decomposition
is generally an approximation. There exist several propagation rules to
determine the relevance scores. In this package are implemented: simple
rule ("simple"), \eqn{\varepsilon}-rule ("epsilon") and
\eqn{\alpha}-\eqn{\beta}-rule ("alpha_beta").

The R6 class can also be initialized using the \code{\link{run_lrp}} function
as a helper function so that no prior knowledge of R6 classes is required.
}
\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
 #----------------------- Example 1: Torch ----------------------------------
library(torch)

# Create nn_sequential model and data
model <- nn_sequential(
  nn_linear(5, 12),
  nn_relu(),
  nn_linear(12, 2),
  nn_softmax(dim = 2)
)
data <- torch_randn(25, 5)

# Create Converter
converter <- convert(model, input_dim = c(5))

# Apply method LRP with simple rule (default)
lrp <- LRP$new(converter, data)

# You can also use the helper function `run_lrp` for initializing
# an R6 LRP object
lrp <- run_lrp(converter, data)

# Print the result as an array for data point one and two
get_result(lrp)[1:2,,]

# Plot the result for both classes
plot(lrp, output_idx = 1:2)

# Plot the boxplot of all datapoints without a preprocess function
boxplot(lrp, output_idx = 1:2, preprocess_FUN = identity)

# ------------------------- Example 2: Neuralnet ---------------------------
if (require("neuralnet")) {
  library(neuralnet)
  data(iris)
  nn <- neuralnet(Species ~ .,
    iris,
    linear.output = FALSE,
    hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
  )

  # Create an converter for this model
  converter <- convert(nn)

  # Create new instance of 'LRP'
  lrp <- run_lrp(converter, iris[, -5], rule_name = "simple")

  # Get the result as an array for data point one and two
  get_result(lrp)[1:2,,]

  # Get the result as a torch tensor for data point one and two
  get_result(lrp, type = "torch.tensor")[1:2]

  # Use the alpha-beta rule with alpha = 2
  lrp <- run_lrp(converter, iris[, -5],
    rule_name = "alpha_beta",
    rule_param = 2
  )

  # Include the last activation into the calculation
  lrp <- run_lrp(converter, iris[, -5],
    rule_name = "alpha_beta",
    rule_param = 2,
    ignore_last_act = FALSE
  )

  # Plot the result for all classes
  plot(lrp, output_idx = 1:3)
}
\dontshow{\}) # examplesIf}
\dontshow{if (torch::torch_is_installed() & Sys.getenv("INNSIGHT_EXAMPLE_KERAS", unset = 0) == 1) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
# ------------------------- Example 3: Keras -------------------------------
if (require("keras") & keras::is_keras_available()) {
  library(keras)

  # Make sure keras is installed properly
  is_keras_available()

  data <- array(rnorm(10 * 60 * 3), dim = c(10, 60, 3))

  model <- keras_model_sequential()
  model \%>\%
    layer_conv_1d(
      input_shape = c(60, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "valid") \%>\%
    layer_conv_1d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "same") \%>\%
    layer_conv_1d(
      kernel_size = 4, filters = 2, activation = "relu",
      padding = "valid") \%>\%
    layer_flatten() \%>\%
    layer_dense(units = 64, activation = "relu") \%>\%
    layer_dense(units = 16, activation = "relu") \%>\%
    layer_dense(units = 3, activation = "softmax")

  # Convert the model
  converter <- convert(model)

  # Apply the LRP method with the epsilon rule for the dense layers and
  # the alpha-beta rule for the convolutional layers
  lrp_comp <- run_lrp(converter, data,
    channels_first = FALSE,
    rule_name = list(Dense_Layer = "epsilon", Conv1D_Layer = "alpha_beta"),
    rule_param = list(Dense_Layer = 0.1, Conv1D_Layer = 1)
  )

  # Plot the result for the first datapoint and all classes
  plot(lrp_comp, output_idx = 1:3)

  # Plot the result as boxplots for first two classes
  boxplot(lrp_comp, output_idx = 1:2)
}
\dontshow{\}) # examplesIf}
\dontshow{if (torch::torch_is_installed() & Sys.getenv("RENDER_PLOTLY", unset = 0) == 1) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
#------------------------- Plotly plots ------------------------------------
if (require("plotly")) {
  # You can also create an interactive plot with plotly.
  # This is a suggested package, so make sure that it is installed
  library(plotly)

  # Result as boxplots
  boxplot(lrp, as_plotly = TRUE)

  # Result of the second data point
  plot(lrp, data_idx = 2, as_plotly = TRUE)
}
\dontshow{\}) # examplesIf}
}
\references{
S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
p. 1-46
}
\seealso{
Other methods: 
\code{\link{ConnectionWeights}},
\code{\link{DeepLift}},
\code{\link{DeepSHAP}},
\code{\link{ExpectedGradient}},
\code{\link{Gradient}},
\code{\link{IntegratedGradient}},
\code{\link{LIME}},
\code{\link{SHAP}},
\code{\link{SmoothGrad}}
}
\concept{methods}
\section{Super class}{
\code{\link[innsight:InterpretingMethod]{innsight::InterpretingMethod}} -> \code{LRP}
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{rule_name}}{(\code{character(1)} or \code{list})\cr
The name of the rule with which the relevance scores
are calculated. Implemented are \code{"simple"}, \code{"epsilon"}, \code{"alpha_beta"}
(and \code{"pass"} but only for 'BatchNorm_Layer'). However, this value
can also be a named list that assigns one of these three rules to each
implemented layer type separately, e.g.,
\code{list(Dense_Layer = "simple", Conv2D_Layer = "alpha_beta")}.
Layers not specified in this list then use the default value \code{"simple"}.
The implemented layer types are:
\tabular{lll}{
\eqn{\cdot} 'Dense_Layer' \tab \eqn{\cdot} 'Conv1D_Layer' \tab \eqn{\cdot} 'Conv2D_Layer'\cr
\eqn{\cdot} 'BatchNorm_Layer' \tab \eqn{\cdot} 'AvgPool1D_Layer' \tab \eqn{\cdot} 'AvgPool2D_Layer'\cr
\eqn{\cdot} 'MaxPool1D_Layer' \tab \eqn{\cdot} 'MaxPool2D_Layer' \tab
}}

\item{\code{rule_param}}{(\code{numeric} or \code{list})\cr
The parameter of the selected rule. Similar to the
argument \code{rule_name}, this can also be a named list that assigns a
rule parameter to each layer type.\cr}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-LRP-new}{\code{LRP$new()}}
\item \href{#method-LRP-clone}{\code{LRP$clone()}}
}
}
\if{html}{\out{
<details open><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="innsight" data-topic="InterpretingMethod" data-id="get_result"><a href='../../innsight/html/InterpretingMethod.html#method-InterpretingMethod-get_result'><code>innsight::InterpretingMethod$get_result()</code></a></span></li>
<li><span class="pkg-link" data-pkg="innsight" data-topic="InterpretingMethod" data-id="plot"><a href='../../innsight/html/InterpretingMethod.html#method-InterpretingMethod-plot'><code>innsight::InterpretingMethod$plot()</code></a></span></li>
<li><span class="pkg-link" data-pkg="innsight" data-topic="InterpretingMethod" data-id="plot_global"><a href='../../innsight/html/InterpretingMethod.html#method-InterpretingMethod-plot_global'><code>innsight::InterpretingMethod$plot_global()</code></a></span></li>
<li><span class="pkg-link" data-pkg="innsight" data-topic="InterpretingMethod" data-id="print"><a href='../../innsight/html/InterpretingMethod.html#method-InterpretingMethod-print'><code>innsight::InterpretingMethod$print()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-LRP-new"></a>}}
\if{latex}{\out{\hypertarget{method-LRP-new}{}}}
\subsection{Method \code{new()}}{
Create a new instance of the \code{LRP} R6 class. When initialized,
the method \emph{LRP} is applied to the given data and the results are stored in
the field \code{result}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LRP$new(
  converter,
  data,
  channels_first = TRUE,
  output_idx = NULL,
  output_label = NULL,
  ignore_last_act = TRUE,
  rule_name = "simple",
  rule_param = NULL,
  winner_takes_all = TRUE,
  verbose = interactive(),
  dtype = "float"
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{converter}}{(\code{\link{Converter}})\cr
An instance of the \code{Converter} class that includes the
torch-converted model and some other model-specific attributes. See
\code{\link{Converter}} for details.\cr}

\item{\code{data}}{(\code{\link{array}}, \code{\link{data.frame}}, \code{\link[torch]{torch_tensor}} or \code{list})\cr
The data to which the method is to be applied. These must
have the same format as the input data of the passed model to the
converter object. This means either
\itemize{
\item an \code{array}, \code{data.frame}, \code{torch_tensor} or array-like format of
size \emph{(batch_size, dim_in)}, if e.g., the model has only one input layer, or
\item a \code{list} with the corresponding input data (according to the
upper point) for each of the input layers.\cr
}}

\item{\code{channels_first}}{(\code{logical(1)})\cr
The channel position of the given data (argument
\code{data}). If \code{TRUE}, the channel axis is placed at the second position
between the batch size and the rest of the input axes, e.g.,
\code{c(10,3,32,32)} for a batch of ten images with three channels and a
height and width of 32 pixels. Otherwise (\code{FALSE}), the channel axis
is at the last position, i.e., \code{c(10,32,32,3)}. If the data
has no channel axis, use the default value \code{TRUE}.\cr}

\item{\code{output_idx}}{(\code{integer}, \code{list} or \code{NULL})\cr
These indices specify the output nodes for which
the method is to be applied. In order to allow models with multiple
output layers, there are the following possibilities to select
the indices of the output nodes in the individual output layers:
\itemize{
\item An \code{integer} vector of indices: If the model has only one output
layer, the values correspond to the indices of the output nodes, e.g.,
\code{c(1,3,4)} for the first, third and fourth output node. If there are
multiple output layers, the indices of the output nodes from the first
output layer are considered.
\item A \code{list} of \code{integer} vectors of indices: If the method is to be
applied to output nodes from different layers, a list can be passed
that specifies the desired indices of the output nodes for each
output layer. Unwanted output layers have the entry \code{NULL} instead of
a vector of indices, e.g., \code{list(NULL, c(1,3))} for the first and
third output node in the second output layer.
\item \code{NULL} (default): The method is applied to all output nodes in
the first output layer but is limited to the first ten as the
calculations become more computationally expensive for more output
nodes.\cr
}}

\item{\code{output_label}}{(\code{character}, \code{factor}, \code{list} or \code{NULL})\cr
These values specify the output nodes for which
the method is to be applied. Only values that were previously passed with
the argument \code{output_names} in the \code{converter} can be used. In order to
allow models with multiple
output layers, there are the following possibilities to select
the names of the output nodes in the individual output layers:
\itemize{
\item A \code{character} vector or \code{factor} of labels: If the model has only one output
layer, the values correspond to the labels of the output nodes named in the
passed \code{Converter} object, e.g.,
\code{c("a", "c", "d")} for the first, third and fourth output node if the
output names are \code{c("a", "b", "c", "d")}. If there are
multiple output layers, the names of the output nodes from the first
output layer are considered.
\item A \code{list} of \code{charactor}/\code{factor} vectors of labels: If the method is to be
applied to output nodes from different layers, a list can be passed
that specifies the desired labels of the output nodes for each
output layer. Unwanted output layers have the entry \code{NULL} instead of
a vector of labels, e.g., \code{list(NULL, c("a", "c"))} for the first and
third output node in the second output layer.
\item \code{NULL} (default): The method is applied to all output nodes in
the first output layer but is limited to the first ten as the
calculations become more computationally expensive for more output
nodes.\cr
}}

\item{\code{ignore_last_act}}{(\code{logical(1)})\cr
Set this logical value to include the last
activation functions for each output layer, or not (default: \code{TRUE}).
In practice, the last activation (especially for softmax activation) is
often omitted.\cr}

\item{\code{rule_name}}{(\code{character(1)} or \code{list})\cr
The name of the rule with which the relevance scores
are calculated. Implemented are \code{"simple"}, \code{"epsilon"}, \code{"alpha_beta"}.
You can pass one of the above characters to apply this rule to all
possible layers. However, this value can also be a named list that
assigns one of these three rules to each
implemented layer type separately, e.g.,
\code{list(Dense_Layer = "simple", Conv2D_Layer = "alpha_beta")}.
Layers not specified in this list then use the default value \code{"simple"}.
The implemented layer types are:\cr
\tabular{lll}{
\eqn{\cdot} 'Dense_Layer' \tab \eqn{\cdot} 'Conv1D_Layer' \tab \eqn{\cdot} 'Conv2D_Layer'\cr
\eqn{\cdot} 'BatchNorm_Layer' \tab \eqn{\cdot} 'AvgPool1D_Layer' \tab \eqn{\cdot} 'AvgPool2D_Layer'\cr
\eqn{\cdot} 'MaxPool1D_Layer' \tab \eqn{\cdot} 'MaxPool2D_Layer' \tab
}
\emph{Note:} For normalization layers like 'BatchNorm_Layer', the rule
\code{"pass"} is implemented as well, which ignores such layers in the
backward pass.\cr}

\item{\code{rule_param}}{(\code{numeric(1)} or \code{list})\cr
The parameter of the selected rule. Note: Only the
rules \code{"epsilon"} and \code{"alpha_beta"} take use of the
parameter. Use the default value \code{NULL} for the default parameters
("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}). Similar to the
argument \code{rule_name}, this can also be a named list that assigns a
rule parameter to each layer type. If the layer type is not specified
in the named list, the default parameters will be used.\cr}

\item{\code{winner_takes_all}}{(\code{logical(1)})\cr
This logical argument is only relevant for
models with a MaxPooling layer. Since many zeros are produced during
the backward pass due to the selection of the maximum value in the
pooling kernel, another variant is implemented, which treats a
MaxPooling as an AveragePooling layer in the backward pass to overcome
the problem of too many zero relevances. With the default value \code{TRUE},
the whole upper-layer relevance is passed to the maximum value in each
pooling window. Otherwise, if \code{FALSE}, the relevance is distributed equally
among all nodes in a pooling window.\cr}

\item{\code{verbose}}{(\code{logical(1)})\cr
This logical argument determines whether a progress bar is
displayed for the calculation of the method or not. The default value is
the output of the primitive R function \code{\link[=interactive]{interactive()}}.\cr}

\item{\code{dtype}}{(\code{character(1)})\cr
The data type for the calculations. Use
either \code{'float'} for \code{\link[torch]{torch_float}} or \code{'double'} for
\code{\link[torch]{torch_double}}.\cr}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A new instance of the R6 class \code{LRP}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-LRP-clone"></a>}}
\if{latex}{\out{\hypertarget{method-LRP-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LRP$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
