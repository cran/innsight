% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Converter.R
\name{Converter}
\alias{Converter}
\title{Converter of an artificial neural network}
\description{
This class analyzes a passed neural network and stores its internal
structure and the individual layers by converting the entire network into an
\code{\link[torch]{nn_module}}. With the help of this converter, many
methods for interpreting the behavior of neural networks are provided, which
give a better understanding of the whole model or individual predictions.
You can use models from the following libraries:
\itemize{
\item \code{torch} (\code{\link[torch]{nn_sequential}})
\item \code{\link[keras]{keras}} (\code{\link[keras]{keras_model}},
\code{\link[keras]{keras_model_sequential}}),
\item \code{\link[neuralnet]{neuralnet}}
}

Furthermore, a model can be passed as a list (see
\code{vignette("detailed_overview", package = "innsight")} or the
\href{https://bips-hb.github.io/innsight/articles/detailed_overview.html#model-as-named-list}{website}).

The R6 class can also be initialized using the \code{\link{convert}} function
as a helper function so that no prior knowledge of R6 classes is required.
}
\details{
In order to better understand and analyze the prediction of a neural
network, the preactivation or other information of the individual layers,
which are not stored in an ordinary forward pass, are often required. For
this reason, a given neural network is converted into a torch-based neural
network, which provides all the necessary information for an interpretation.
The converted torch model is stored in the field \code{model} and is an instance
of \code{\link[innsight:ConvertedModel]{ConvertedModel}}.
However, before the torch model is created, all relevant details of the
passed model are extracted into a named list. This list can be saved
in complete form in the \code{model_as_list} field with the argument
\code{save_model_as_list}, but this may consume a lot of memory for large
networks and is not done by default. Also, this named list can again be
used as a passed model for the class \code{Converter}, which will be described
in more detail in the section 'Implemented Libraries'.
\subsection{Implemented methods}{

An object of the Converter class can be applied to the
following methods:
\itemize{
\item \emph{Layerwise Relevance Propagation} (\link{LRP}), Bach et al. (2015)
\item \emph{Deep Learning Important Features} (\link{DeepLift}), Shrikumar et al. (2017)
\item \emph{\link{DeepSHAP}}, Lundberg et al. (2017)
\item \emph{\link{SmoothGrad}} including \emph{SmoothGrad\eqn{\times}Input}, Smilkov et al. (2017)
\item \emph{Vanilla \link{Gradient}} including \emph{Gradient\eqn{\times}Input}
\item \emph{Integrated gradients} (\link{IntegratedGradient}), Sundararajan et al. (2017)
\item \emph{Expected gradients} (\link{ExpectedGradient}), Erion et al. (2021)
\item \emph{\link{ConnectionWeights}}, Olden et al. (2004)
\item \emph{Local interpretable model-agnostic explanation (\link{LIME})}, Ribeiro et al. (2016)
\item \emph{Shapley values (\link{SHAP})}, Lundberg et al. (2017)
}
}

\subsection{Implemented libraries}{

The converter is implemented for models from the libraries
\code{\link[torch]{nn_sequential}},
\code{\link[neuralnet]{neuralnet}} and \code{\link[keras]{keras}}. But you
can also write a wrapper for other libraries because a model can be passed
as a named list which is described in detail in the vignette "In-depth
explanation" \cr
(see \code{vignette("detailed_overview", package = "innsight")} or the
\href{https://bips-hb.github.io/innsight/articles/detailed_overview.html#model-as-named-list}{website}).
}
}
\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
#----------------------- Example 1: Torch ----------------------------------
library(torch)

model <- nn_sequential(
  nn_linear(5, 10),
  nn_relu(),
  nn_linear(10, 2, bias = FALSE),
  nn_softmax(dim = 2)
)
data <- torch_randn(25, 5)

# Convert the model (for torch models is 'input_dim' required!)
converter <- Converter$new(model, input_dim = c(5))

# You can also use the helper function `convert()` for initializing a
# Converter object
converter <- convert(model, input_dim = c(5))

# Get the converted model stored in the field 'model'
converted_model <- converter$model

# Test it with the original model
mean(abs(converted_model(data)[[1]] - model(data)))


#----------------------- Example 2: Neuralnet ------------------------------
if (require("neuralnet")) {
  library(neuralnet)
  data(iris)

  # Train a neural network
  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
    iris,
    linear.output = FALSE,
    hidden = c(3, 2), act.fct = "tanh", rep = 1
  )

  # Convert the model
  converter <- convert(nn)

  # Print all the layers
  converter$model$modules_list
}
\dontshow{\}) # examplesIf}
\dontshow{if (torch::torch_is_installed() & Sys.getenv("INNSIGHT_EXAMPLE_KERAS", unset = 0) == 1) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
#----------------------- Example 3: Keras ----------------------------------
if (require("keras") & keras::is_keras_available()) {
  library(keras)

  # Make sure keras is installed properly
  is_keras_available()

  # Define a keras model
  model <- keras_model_sequential() \%>\%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "relu", padding = "same") \%>\%
    layer_conv_2d(
      kernel_size = 8, filters = 4,
      activation = "tanh", padding = "same") \%>\%
    layer_conv_2d(
      kernel_size = 4, filters = 2,
      activation = "relu", padding = "same") \%>\%
    layer_flatten() \%>\%
    layer_dense(units = 64, activation = "relu") \%>\%
    layer_dense(units = 1, activation = "sigmoid")

  # Convert this model and save model as list
  converter <- convert(model, save_model_as_list = TRUE)

  # Print the converted model as a named list
  str(converter$model_as_list, max.level = 1)
}
\dontshow{\}) # examplesIf}
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
#----------------------- Example 4: List  ----------------------------------

# Define a model

model <- list()
model$input_dim <- 5
model$input_names <- list(c("Feat1", "Feat2", "Feat3", "Feat4", "Feat5"))
model$input_nodes <- c(1)
model$output_dim <- 2
model$output_names <- list(c("Cat", "no-Cat"))
model$output_nodes <- c(2)
model$layers$Layer_1 <-
  list(
    type = "Dense",
    weight = matrix(rnorm(5 * 20), 20, 5),
    bias = rnorm(20),
    activation_name = "tanh",
    dim_in = 5,
    dim_out = 20,
    input_layers = 0, # '0' means model input layer
    output_layers = 2
  )
model$layers$Layer_2 <-
  list(
    type = "Dense",
    weight = matrix(rnorm(20 * 2), 2, 20),
    bias = rnorm(2),
    activation_name = "softmax",
    input_layers = 1,
    output_layers = -1 # '-1' means model output layer
    #dim_in = 20, # These values are optional, but
    #dim_out = 2  # useful for internal checks
  )

# Convert the model
converter <- convert(model)

# Get the model as a torch::nn_module
torch_model <- converter$model

# You can use it as a normal torch model
x <- torch::torch_randn(3, 5)
torch_model(x)
\dontshow{\}) # examplesIf}
}
\references{
\itemize{
\item J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
quantifying variable importance in artificial neural networks using
simulated data.} Ecological Modelling 178, p. 389–397
\item S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
p. 1-46
\item M. T. Ribeiro et al. (2016) \emph{"Why should I trust you?": Explaining
the predictions of any classifier.} KDD 2016, p. 1135-1144
\item A. Shrikumar et al. (2017) \emph{Learning important features through
propagating activation differences.}  ICML 2017, p. 4844-4866
\item D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
CoRR, abs/1706.03825
M. Sundararajan et al. (2017) \emph{Axiomatic attribution for deep networks.}
ICML 2017, p.3319-3328
\item S. Lundberg et al. (2017) \emph{A unified approach to interpreting model
predictions.} NIPS 2017, p. 4768-4777
\item G. Erion et al. (2021) \emph{Improving performance of deep learning models
with axiomatic attribution priors and expected gradients.} Nature Machine
Intelligence 3, p. 620-631
}
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{model}}{(\code{\link{ConvertedModel}})\cr
The converted neural network based on the torch module \link{ConvertedModel}.\cr}

\item{\code{input_dim}}{(\code{list})\cr
A list of the input dimensions of each input layer. Since
internally the "channels first" format is used for all calculations, the
input shapes are already in this format. In addition, the batch
dimension isn't included, e.g., for an input layer of shape \verb{c(*,32,32,3)}
with channels in the last axis you get \code{list(c(3,32,32))}.\cr}

\item{\code{input_names}}{(\code{list})\cr
A list with the names as factors for each input
dimension of the shape as stored in the field \code{input_dim}.\cr}

\item{\code{output_dim}}{(\code{list})\cr
A list of the output dimensions of each output layer.\cr}

\item{\code{output_names}}{(\code{list})\cr A list with the names as factors for each
output dimension of shape as stored in the field \code{output_dim}.\cr}

\item{\code{model_as_list}}{(\code{list})\cr
The model stored in a named list (see details for more
information). By default, the entry \code{model_as_list$layers} is deleted
because it may require a lot of memory for large networks. However, with
the argument \code{save_model_as_list} this can be saved anyway.\cr}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-Converter-new}{\code{Converter$new()}}
\item \href{#method-Converter-print}{\code{Converter$print()}}
\item \href{#method-Converter-clone}{\code{Converter$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Converter-new"></a>}}
\if{latex}{\out{\hypertarget{method-Converter-new}{}}}
\subsection{Method \code{new()}}{
Create a new \link{Converter} object for a given neural network. When initialized,
the model is inspected, converted as a list and then the a
torch-converted model (\link{ConvertedModel}) is created and stored in
the field \code{model}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Converter$new(
  model,
  input_dim = NULL,
  input_names = NULL,
  output_names = NULL,
  dtype = "float",
  save_model_as_list = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{model}}{(\code{\link[torch]{nn_sequential}},
\code{\link[keras]{keras_model}},
\code{\link[neuralnet]{neuralnet}} or \code{list})\cr
A trained neural network for classification or regression
tasks to be interpreted. Only models from the following types or
packages are allowed: \code{\link[torch]{nn_sequential}},
\code{\link[keras]{keras_model}},
\code{\link[keras]{keras_model_sequential}},
\code{\link[neuralnet]{neuralnet}} or a named list (see details).}

\item{\code{input_dim}}{(\code{integer} or \code{list})\cr
The model input dimension excluding the batch
dimension. If there is only one input layer it can be specified as
a vector, otherwise use a list of the shapes of the
individual input layers.\cr
\emph{Note:} This argument is only necessary for \code{torch::nn_sequential},
for all others it is automatically extracted from the passed model
and used for internal checks. In addition, the input dimension
\code{input_dim} has to be in the format "channels first".\cr}

\item{\code{input_names}}{(\code{character}, \code{factor} or \code{list})\cr
The input names of the model excluding the batch dimension. For a model
with a single input layer and input axis (e.g., for tabular data), the
input names can be specified as a character vector or factor, e.g.,
for a dense layer with 3 input features use \code{c("X1", "X2", "X3")}. If
the model input consists of multiple axes (e.g., for signal and
image data), use a list of character vectors or factors for each axis
in the format "channels first", e.g., use
\code{list(c("C1", "C2"), c("L1","L2","L3","L4","L5"))} for a 1D
convolutional input layer with signal length 4 and 2 channels. For
models with multiple input layers, use a list of the upper ones for each
layer.\cr
\emph{Note:} This argument is optional and otherwise the names are
generated automatically. But if this argument is set, all found
input names in the passed model will be disregarded.\cr}

\item{\code{output_names}}{(\code{character}, \code{factor} or \code{list})\cr
A character vector with the names for the output dimensions
excluding the batch dimension, e.g., for a model with 3 output nodes use
\code{c("Y1", "Y2", "Y3")}. Instead of a character
vector you can also use a factor to set an order for the plots. If the
model has multiple output layers, use a list of the upper ones.\cr
\emph{Note:} This argument is optional and otherwise the names are
generated automatically. But if this argument is set, all found
output names in the passed model will be disregarded.\cr}

\item{\code{dtype}}{(\code{character(1)})\cr
The data type for the calculations. Use
either \code{'float'} for \link[torch:torch_dtype]{torch::torch_float} or \code{'double'} for
\link[torch:torch_dtype]{torch::torch_double}.\cr}

\item{\code{save_model_as_list}}{(\code{logical(1)})\cr
This logical value specifies whether the
passed model should be stored as a list. This list can take
a lot of memory for large networks, so by default the model is not
stored as a list (\code{FALSE}).\cr}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A new instance of the R6 class \code{Converter}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Converter-print"></a>}}
\if{latex}{\out{\hypertarget{method-Converter-print}{}}}
\subsection{Method \code{print()}}{
Print a summary of the \code{Converter} object. This summary contains the
individual fields and in particular the torch-converted model
(\link{ConvertedModel}) with the layers.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Converter$print()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns the \code{Converter} object invisibly via \code{\link[base:invisible]{base::invisible}}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Converter-clone"></a>}}
\if{latex}{\out{\hypertarget{method-Converter-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Converter$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
