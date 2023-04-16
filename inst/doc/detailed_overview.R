## ---- include = FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(
  fig.dpi = ifelse(Sys.getenv("RENDER_PLOTLY", unset = 0) == 1, 400, 50),
  collapse = TRUE,
  eval = torch::torch_is_installed(),
  comment = "#>",
  fig.align = "center",
  out.width = "90%"
)

library(innsight)

## ---- eval = FALSE----------------------------------------------------------------------
#  converter <- Converter$new(model,
#    input_dim = NULL,
#    input_names = NULL,
#    output_names = NULL,
#    dtype = "float",
#    save_model_as_list = FALSE
#  )

## ---------------------------------------------------------------------------------------
library(torch)

torch_model <- nn_sequential(
  nn_conv2d(3, 5, c(2, 2), stride = 2, padding = 3),
  nn_relu(),
  nn_avg_pool2d(c(2, 2)),
  nn_flatten(),
  nn_linear(80, 32),
  nn_relu(),
  nn_dropout(),
  nn_linear(32, 2),
  nn_softmax(dim = 2)
)

# For torch models the optional argument `input_dim` becomes a necessary one
converter <- Converter$new(torch_model, input_dim = c(3, 10, 10))

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
library(keras)

# Create model
keras_model_seq <- keras_model_sequential()
keras_model_seq <- keras_model_seq %>%
  layer_dense(10, input_shape = c(5), activation = "softplus") %>%
  layer_dense(8, use_bias = FALSE, activation = "tanh") %>%
  layer_dropout(0.2) %>%
  layer_dense(4, activation = "softmax")

converter <- Converter$new(keras_model_seq)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
library(keras)

input_image <- layer_input(shape = c(10, 10, 3))
input_tab <- layer_input(shape = c(20))

conv_part <- input_image %>%
  layer_conv_2d(5, c(2, 2), activation = "relu", padding = "same") %>%
  layer_average_pooling_2d() %>%
  layer_conv_2d(4, c(2, 2)) %>%
  layer_activation(activation = "softplus") %>%
  layer_flatten()

output <- layer_concatenate(list(conv_part, input_tab)) %>%
  layer_dense(50, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(3, activation = "softmax")

keras_model_concat <- keras_model(inputs = list(input_image, input_tab), outputs = output)

converter <- Converter$new(keras_model_concat)

## ---------------------------------------------------------------------------------------
library(neuralnet)
data(iris)
set.seed(42)

# Create model
neuralnet_model <- neuralnet(Species ~ Petal.Length + Petal.Width, iris,
  linear.output = FALSE
)

# Convert model
converter <- Converter$new(neuralnet_model)
# Show input names
converter$input_names
# Show output names
converter$output_names

## ---- eval = FALSE----------------------------------------------------------------------
#  model$input_dim <- c(5)

## ---- eval = FALSE----------------------------------------------------------------------
#  input_dim <- list(c(10), c(3, 10, 10)) # channels have to be first!

## ---- eval = FALSE----------------------------------------------------------------------
#  model$input_names <- c("Feature_1", "Feature_2")

## ---- eval = FALSE----------------------------------------------------------------------
#  model$input_names <- list(c("C1", "C2"), c("H1", "H2"), c("W1", "W2"))

## ---- eval = FALSE----------------------------------------------------------------------
#  model$input_names <- list(
#    list(c("Feature_1", "Feature_2")),
#    list(c("C1", "C2"), c("H1", "H2"), c("W1", "W2"))
#  )

## ---------------------------------------------------------------------------------------
# Define dense layer
dense_layer <- list(
  type = "Dense",
  input_layers = 0, # '0' means input layer
  output_layers = 2,
  weight = matrix(rnorm(5 * 2), 2, 5),
  bias = rnorm(2),
  activation_name = "tanh",
  dim_in = 5, # optional
  dim_out = 2 # optional
)

## ---- eval = FALSE----------------------------------------------------------------------
#  # 1D convolutional layer
#  conv_1D <- list(
#    type = "Conv1D",
#    input_layers = 1,
#    output_layers = 3,
#    weight = array(rnorm(8 * 3 * 2), dim = c(8, 3, 2)),
#    bias = rnorm(8),
#    padding = c(2, 1),
#    activation_name = "tanh",
#    dim_in = c(3, 10), # optional
#    dim_out = c(8, 9) # optional
#  )
#  
#  # 2D convolutional layer
#  conv_2D <- list(
#    type = "Conv2D",
#    input_layes = 3,
#    output_layers = 5,
#    weight = array(rnorm(8 * 3 * 2 * 4), dim = c(8, 3, 2, 4)),
#    bias = rnorm(8),
#    padding = c(1, 1, 0, 0),
#    dilation = c(1, 2),
#    activation_name = "relu",
#    dim_in = c(3, 10, 10) # optional
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # 2D average pooling layer
#  avg_pool2D <- list(
#    type = "AveragePooling2D",
#    input_layers = 1,
#    output_layers = 3,
#    kernel_size = c(2, 2)
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # batch normalization layer
#  batchnorm <- list(
#    type = "BatchNorm",
#    input_layers = 1,
#    output_layers = 3,
#    num_features = 3,
#    eps = 1e-4,
#    gamma = c(1.1, 0.0, -0.3),
#    beta = c(1, -3, -1.4),
#    run_mean = c(-1.9, 3, 2.3),
#    run_var = c(1, 2.1, 3.5)
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # flatten layer
#  flatten <- list(
#    type = "Flatten",
#    input_layers = 1,
#    output_layers = 3,
#    start_dim = 2, # optional
#    end_dim = 4, # optional
#    dim_in = c(3, 10, 10), # optional
#    out_dim = c(300) # optional
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # global MaxPooling layer
#  global_max_pool2D <- list(
#    type = "GlobalPooling",
#    input_layers = 1,
#    output_layers = 3,
#    method = "max",
#    dim_in = c(3, 10, 10), # optional
#    out_dim = c(3) # optional
#  )
#  
#  # global AvgPooling layer
#  global_avg_pool1D <- list(
#    type = "GlobalPooling",
#    input_layers = 1,
#    output_layers = 3,
#    method = "average",
#    dim_in = c(3, 10), # optional
#    out_dim = c(3) # optional
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # padding layer
#  padding <- list(
#    type = "Padding",
#    input_layers = 1,
#    output_layers = 3,
#    padding = c(2, 4),
#    mode = "constant",
#    value = 1,
#    dim_in = c(3, 10), # optional
#    out_dim = c(3, 16) # optional
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # concatenation layer
#  concat <- list(
#    type = "Concatenation",
#    input_layers = c(1, 3),
#    output_layers = 5,
#    dim = 2,
#    dim_in = list(c(5), c(3)), # optional
#    out_dim = c(8) # optional
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  # adding layer
#  add <- list(
#    type = "Add",
#    input_layers = c(1, 3),
#    output_layers = 5,
#    dim_in = list(c(3, 10, 10), c(3, 10, 10)), # optional
#    out_dim = c(3, 10, 10) # optional
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  list(c("set", "your", "labels", "here!"))
#  # or as a factor
#  list(
#    factor(c("set", "your", "labels", "here"),
#      levels = c("labels", "set", "your", "here")
#    )
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  list(
#    c("channels", "are", "first"),
#    c("Length1", "Length2", "Length3", "Length4", "Length5", "Length6")
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  list(
#    c("channels", "are", "first"),
#    c("then", "comes", "the", "image height"),
#    c("and", "then", "the", "width")
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  list(
#    # first input layer
#    list(c("Feat_1", "Feat_2", "Feat_3", "Feat_4")),
#    # second input layer
#    list(
#      c("C1", "C2", "C3"),
#      c("Height_1", "Height_2", "Height_3", "Height_4"),
#      c("W1", "W2", "W3", "W4")
#    )
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  c("First output node", "second one", "last output node")
#  # or as a factor
#  factor(c("First output node", "second one", "last output node"),
#    levels = c("last output node", "First output node", "second one", )
#  )

## ---- eval = FALSE----------------------------------------------------------------------
#  list(
#    c("Yes", "No!"),
#    c("Out1", "Out2", "Out3", "Out4")
#  )

## ---------------------------------------------------------------------------------------
torch_manual_seed(123)
A <- torch_randn(10, 10)
B <- torch_randn(10, 10)

## ---------------------------------------------------------------------------------------
# result of first row and first column after matrix multiplication
res1 <- torch_mm(A, B)[1, 1]
# calculation by hand
res2 <- sum(A[1, ] * B[, 1])

# difference:
res1 - res2

## ---------------------------------------------------------------------------------------
torch_manual_seed(123)
A <- torch_randn(10, 10, dtype = torch_double())
B <- torch_randn(10, 10, dtype = torch_double())

# result of first row and first column after matrix multiplication
res1 <- torch_mm(A, B)[1, 1]
# calculation by hand
res2 <- sum(A[1, ] * B[, 1])

# difference:
res1 - res2

## ---------------------------------------------------------------------------------------
# Convert the model and save the model as a list
converter <- Converter$new(keras_model_concat, save_model_as_list = TRUE)

# Get the field `input_dim`
converter$input_dim

## ---------------------------------------------------------------------------------------
# create input in the format "channels last"
x <- list(
  array(rnorm(3 * 10 * 10), dim = c(1, 10, 10, 3)),
  array(rnorm(20), dim = c(1, 20))
)

# output of the original model
y_true <- as.array(keras_model_concat(x))
# output of the torch-converted model (the data 'x' is in the format channels
# last, hence we need to set the argument 'channels_first = FALSE')
y <- as.array(converter$model(x, channels_first = FALSE)[[1]])

# mean squared error
mean((y - y_true)**2)

## ---------------------------------------------------------------------------------------
# get the calculated output dimension
str(converter$output_dim)
# get the generated output names (one layer with three output nodes)
str(converter$output_names)
# get the generated input names
str(converter$input_names)

## ---------------------------------------------------------------------------------------
# get the mode as a list
model_as_list <- converter$model_as_list
# print the fourth layer
str(model_as_list$layers[[4]])
# let's change the activation function to "relu"
model_as_list$layers[[4]]$activation_name <- "relu"
# create a Converter object with the modified model
converter_modified <- Converter$new(model_as_list)

# now, we get different results for the same input because of the relu activation
converter_modified$model(x, channels_first = FALSE)
converter$model(x, channels_first = FALSE)

## ---- echo=FALSE------------------------------------------------------------------------
options(width = 90)

## ---------------------------------------------------------------------------------------
# print the Converter instance
converter

## ---- eval = FALSE----------------------------------------------------------------------
#  # Apply the selected method
#  method <- Method$new(converter, data,
#    channels_first = TRUE,
#    output_idx = NULL,
#    ignore_last_act = TRUE,
#    verbose = interactive(),
#    dtype = "float"
#  )

## ---- eval = TRUE, echo=FALSE, fig.cap = "**Fig. 1:** Example neural network", out.width = "80%"----
knitr::include_graphics("images/example_net.png")

## ---------------------------------------------------------------------------------------
model <- list(
  input_dim = 1,
  input_nodes = 1,
  input_names = c("x"),
  output_nodes = 2,
  output_names = c("y"),
  layers = list(
    list(
      type = "Dense",
      input_layers = 0,
      output_layers = 2,
      weight = matrix(c(1, 0.8, 2), nrow = 3),
      bias = c(0, -0.4, -1.2),
      activation_name = "relu"
    ),
    list(
      type = "Dense",
      input_layers = 1,
      output_layers = -1,
      weight = matrix(c(1, -1, 1), nrow = 1),
      bias = c(0),
      activation_name = "tanh"
    )
  )
)

converter <- Converter$new(model)

## ---- eval = FALSE----------------------------------------------------------------------
#  grad <- Gradient$new(converter, data,
#    times_input = FALSE,
#    ... # other arguments inherited from 'InterpretingMethod'
#  )

## ---- echo = FALSE, fig.width=5, fig.height= 3, fig.cap= "**Fig. 2:** Gradient method"----
library(ggplot2)

func <- function(x) {
  as.array(converter$model(torch_tensor(matrix(x, ncol = 1)))[[1]])
}

grad_func <- function(x) {
  grad <- x
  grad <- ifelse(x <= 0, 0, grad)
  grad <- ifelse(x > 0 & x <= 0.5, 1 / cosh(x)**2, grad)
  grad <- ifelse(x > 0.5 & x <= 0.6, 0.2 / cosh(0.2 * x + 0.4)**2, grad)
  grad <- ifelse(x > 0.6, 2.2 / cosh(0.8 - 2.2 * x)**2, grad)

  grad
}

base <-
  ggplot() +
  xlim(-0.2, 1.3) +
  ylim(-0.2, 1) +
  xlab("x") +
  geom_vline(aes(xintercept = 0)) +
  geom_hline(aes(yintercept = 0)) +
  annotate("text", label = "f", x = 0.92, y = 0.95, size = 6)

base +
  geom_function(fun = func, alpha = 0.7) +
  geom_segment(aes(x = 0.45, y = -0.05, xend = 0.45, yend = 0.05), linewidth = 0.8) +
  geom_segment(aes(x = -0.03, y = tanh(0.45), xend = 0.03, yend = tanh(0.45)), 
               linewidth = 0.8) +
  annotate("text", label = "x[1]", x = 0.45, y = -0.12, size = 5, parse = TRUE) +
  annotate("text", label = "f(x[1])", x = -0.13, y = tanh(0.45), size = 5, parse = TRUE) +
  geom_segment(aes(
    x = 0.25, xend = 0.65, y = tanh(0.45) - 0.2 / cosh(0.45)**2,
    yend = tanh(0.45) + 0.2 / cosh(0.45)**2
  ),
  color = "red", alpha = 0.7, linewidth = 1.5
  ) +
  geom_point(
    data = data.frame(x = 0.45, y = tanh(0.45)),
    mapping = aes(x = x, y = y)
  )

## ---------------------------------------------------------------------------------------
data <- matrix(c(0.45), 1, 1)

# Apply method (but don't ignore last activation)
grad <- Gradient$new(converter, data, ignore_last_act = FALSE)
# get result
grad$get_result()

## ---- eval = FALSE----------------------------------------------------------------------
#  smoothgrad <- SmoothGrad$new(converter, data,
#    n = 50,
#    noise_level = 0.1,
#    times_input = FALSE,
#    ... # other arguments inherited from 'InterpretingMethod'
#  )

## ---- echo = FALSE, fig.width=5, fig.height= 3, fig.cap= "**Fig. 3:** SmoothGrad method"----
set.seed(111)
fig <- base +
  geom_function(fun = func, alpha = 0.7) +
  geom_segment(aes(x = 0.6, y = -0.05, xend = 0.6, yend = 0.05), linewidth = 0.8) +
  geom_segment(aes(x = -0.03, y = func(0.6), xend = 0.03, yend = func(0.6)), linewidth = 0.8) +
  annotate("text", label = "x[1]", x = 0.6, y = -0.12, size = 6, parse = TRUE) +
  annotate("text", label = "f(x[1])", x = -0.13, y = func(0.6), size = 6, parse = TRUE)

eps <- rnorm(10) * 0.2
x0 <- 0.6
y0 <- as.vector(func(x0))
x <- x0 + eps
y <- as.vector(func(x))
grad <- grad_func(x)
norm <- (1 + grad^2)**0.5
grad_mean <- mean(grad)
norm_mean <- (1 + grad_mean**2)**0.5


data <- data.frame(
  x = x - 0.4 / norm, xend = x + 0.4 / norm,
  y = y - grad * 0.4 / norm, yend = y + grad * 0.4 / norm
)
mean_grad <- data.frame(
  x = x0 - 0.6 / norm_mean, xend = x0 + 0.6 / norm_mean,
  y = y0 - grad_mean * 0.6 / norm_mean,
  yend = y0 + grad_mean * 0.6 / norm_mean
)

fig +
  geom_segment(
    data = data, mapping = aes(x = x, xend = xend, y = y, yend = yend),
    color = "darkblue", alpha = 0.3, linewidth = 0.5
  ) +
  geom_segment(
    data = mean_grad, mapping = aes(x = x, xend = xend, y = y, yend = yend),
    color = "red", alpha = 0.9, linewidth = 1.25
  ) +
  geom_function(fun = func, alpha = 0.7) +
  geom_point(mapping = aes(x = x, y = y), color = "blue", size = 0.8) +
  geom_point(
    data = data.frame(x = 0.6, y = func(0.6)),
    mapping = aes(x = x, y = y)
  )

## ---------------------------------------------------------------------------------------
data <- matrix(c(0.6), 1, 1)

# Apply method
smoothgrad <- SmoothGrad$new(converter, data,
  noise_level = 0.2,
  n = 50,
  ignore_last_act = FALSE # include the tanh activation
) 
# get result
smoothgrad$get_result()

## ---- eval = FALSE----------------------------------------------------------------------
#  # the "x Input" variant of method "Gradient"
#  grad_x_input <- Gradient$new(converter, data,
#    times_input = TRUE,
#    ... # other arguments of method "Gradient"
#  )
#  
#  # the "x Input" variant of method "SmoothGrad"
#  smoothgrad_x_input <- SmoothGrad$new(converter, data,
#    times_input = TRUE,
#    ... # other arguments of method "SmoothGrad"
#  )

## ---- echo = FALSE, fig.width=5, fig.height= 3, fig.cap= "**Fig. 4:** Gradient$\\times$Input method"----
base +
  geom_function(fun = func, alpha = 0.7) +
  geom_segment(aes(xend = 0, yend = 0.5 * (c(func(0.49)) - grad_func(0.49) * 0.49), x = 0.75, y = 0.25), color = "black", linewidth = 0.25, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), alpha = 0.8) +
  annotate("text", label = "ε(f,0.49,0)", x = 0.85, y = 0.25) +
  geom_segment(aes(x = 0.49, y = func(0.49), xend = 0.49, yend = grad_func(0.49) * 0.49), color = "red", linewidth = 0.3) +
  geom_segment(aes(x = 0, y = 0, xend = 0, yend = c(func(0.49)) - grad_func(0.49) * 0.49), color = "red", linewidth = 0.3) +
  geom_segment(aes(xend = 0.49, yend = c(func(0.49)) - 0.5 * (c(func(0.49)) - grad_func(0.49) * 0.49), x = 0.75, y = 0.25), color = "black", linewidth = 0.25, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), alpha = 0.8) +
  geom_function(fun = function(x) grad_func(0.49) * (x - 0.49) + c(func(0.49)), color = "red", alpha = 0.7, xlim = c(-0.2, 1.1)) +
  geom_segment(aes(x = 0.49, y = -0.05, xend = 0.49, yend = 0.05), linewidth = 0.8) +
  geom_segment(aes(x = -0.03, y = func(0.49), xend = 0.03, yend = func(0.49)), linewidth = 0.8) +
  annotate("text", label = "x[1]", x = 0.49, y = -0.12, size = 6, parse = TRUE) +
  annotate("text", label = "f(x[1])", x = -0.13, y = func(0.49), size = 6, parse = TRUE) +
  geom_point(
    data = data.frame(x = 0.49, y = grad_func(0.49) * 0.49),
    mapping = aes(x = x, y = y), color = "red", alpha = 0.7
  ) +
  geom_point(
    data = data.frame(x = 0.49, y = func(0.49)),
    mapping = aes(x = x, y = y), color = "black", alpha = 0.7
  )

## ---------------------------------------------------------------------------------------
data <- matrix(c(0.49), 1, 1)

# Apply method
grad_x_input <- Gradient$new(converter, data,
  times_input = TRUE,
  ignore_last_act = FALSE # include the tanh activation
) 
# get result
grad_x_input$get_result()

## ---- echo = FALSE, fig.width=5, fig.height= 3, fig.cap= "**Fig. 5:** SmoothGrad$\\times$Input method"----
set.seed(111)

x <- 0.49 + rnorm(10) * 0.2
m <- grad_func(x)
b <- c(func(x)) - m * x

base +
  geom_function(fun = func, alpha = 0.7) +
  geom_segment(aes(x = 0.49, y = -0.05, xend = 0.49, yend = 0.05), linewidth = 0.8) +
  geom_segment(aes(x = -0.03, y = func(0.49), xend = 0.03, yend = func(0.49)), linewidth = 0.8) +
  annotate("text", label = "x[1]", x = 0.49, y = -0.12, size = 6, parse = TRUE) +
  annotate("text", label = "f(x[1])", x = -0.13, y = func(0.49), size = 6, parse = TRUE) +
  geom_function(fun = function(z) m[1] * z + b[1], color = "blue", alpha = 0.3, na.rm = TRUE, 
                linewidth = 0.3) +
  geom_function(fun = function(z) m[2] * z + b[2], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[3] * z + b[3], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[4] * z + b[4], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[5] * z + b[5], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[6] * z + b[6], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[7] * z + b[7], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[8] * z + b[8], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[9] * z + b[9], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_function(fun = function(z) m[10] * z + b[10], color = "blue", alpha = 0.3, na.rm = TRUE, linewidth = 0.3) +
  geom_point(
    data = data.frame(x = x, y = func(x)), mapping = aes(x = x, y = y), color = "black",
    alpha = 0.3, size = 0.8
  ) +
  geom_point(
    data = data.frame(x = x, y = m * x), mapping = aes(x = x, y = y), color = "blue",
    alpha = 0.3
  ) +
  geom_point(data = data.frame(x = 0.49, y = mean(m * x)), mapping = aes(x = x, y = y), color = "red") +
  geom_point(
    data = data.frame(x = 0.49, y = func(0.49)),
    mapping = aes(x = x, y = y), color = "black", alpha = 0.7
  )

## ---------------------------------------------------------------------------------------
data <- matrix(c(0.49), 1, 1)

# Apply method
smoothgrad_x_input <- SmoothGrad$new(converter, data,
  times_input = TRUE,
  ignore_last_act = FALSE # include the tanh activation
) 
# get result
smoothgrad_x_input$get_result()

## ---- eval = TRUE, echo=FALSE, fig.cap = "**Fig. 6:** Layerwise Relevance Propagation"----
knitr::include_graphics("images/lrp.png")

## ---- eval = FALSE----------------------------------------------------------------------
#  lrp <- LRP$new(converter, data,
#    rule_name = "simple",
#    rule_param = NULL,
#    winner_takes_all = TRUE,
#    ... # other arguments inherited from 'InterpretingMethod'
#  )

## ---------------------------------------------------------------------------------------
# We can analyze multiple inputs simultaneously
data <- matrix(
  c(
    0.49, # only neuron without bias term is activated
    0.6   # neuron with bias term is activated
  ), 
  ncol = 1
)

# Apply LRP with simple rule
lrp <- LRP$new(converter, data,
  ignore_last_act = FALSE
)
lrp$get_result()

# get approximation error
matrix(lrp$get_result()) - as_array(converter$model(torch_tensor(data))[[1]])

## ---- echo = FALSE, fig.width=7, fig.height= 4, fig.cap= "**Fig. 7:** LRP method", warning=FALSE, message=FALSE, results='hide'----


fun_1 <- function(x) {
  LRP$new(converter, matrix(x, ncol = 1), ignore_last_act = FALSE)$get_result()
}

fun_2 <- function(x) {
  LRP$new(converter, matrix(x, ncol = 1), ignore_last_act = FALSE, rule_name = "epsilon", rule_param = 0.1)$get_result()
}

fun_3 <- function(x) {
  LRP$new(converter, matrix(x, ncol = 1), ignore_last_act = FALSE, rule_name = "alpha_beta", rule_param = 0.5)$get_result()
}

fun_4 <- function(x) {
  LRP$new(converter, matrix(x, ncol = 1), ignore_last_act = FALSE, rule_name = "alpha_beta", rule_param = 1)$get_result()
}

ggplot() +
  xlim(-0.2, 1.3) +
  xlab("x") +
  geom_vline(aes(xintercept = 0)) +
  geom_hline(aes(yintercept = 0)) +
  geom_function(data = data.frame(label = "f"), mapping = aes(color = label), color = "black", fun = func, size = 2) +
  geom_function(data = data.frame(label = "simple"), mapping = aes(color = label), fun = fun_1) +
  geom_function(data = data.frame(label = "epsilon"), mapping = aes(color = label), fun = fun_2) +
  geom_function(data = data.frame(label = "alpha_beta (0.5)"), mapping = aes(color = label), fun = fun_3) +
  geom_function(data = data.frame(label = "alpha_beta (1)"), mapping = aes(color = label), fun = fun_4) +
  labs(color = "Rule")

## ---- eval = FALSE----------------------------------------------------------------------
#  deeplift <- DeepLift$new(converter, data,
#    x_ref = NULL,
#    rule_name = "rescale",
#    winner_takes_all = TRUE,
#    ... # other arguments inherited from 'InterpretingMethod'
#  )

## ---------------------------------------------------------------------------------------
# Create data
x <- matrix(c(0.55))
x_ref <- matrix(c(0.1))

# Apply method DeepLift with rescale rule
deeplift <- DeepLift$new(converter, x, x_ref = x_ref, ignore_last_act = FALSE)

# Get result
get_result(deeplift)

## ---------------------------------------------------------------------------------------
library(neuralnet)
set.seed(42)

# Crate model with package 'neuralnet'
model <- neuralnet(Species ~ ., iris, hidden = 5, linear.output = FALSE)

# Step 1: Create 'Converter'
conv <- Converter$new(model)

# Step 2: Apply DeepLift (reveal-cancel rule)
x_ref <- matrix(colMeans(iris[, -5]), nrow = 1) # use colmeans as reference value
deeplift <- DeepLift$new(conv, iris[, -5],
  x_ref = x_ref, ignore_last_act = FALSE,
  rule_name = "reveal_cancel"
)

# Verify exact decomposition
y <- predict(model, iris[, -5])
y_ref <- predict(model, x_ref[rep(1, 150), ])
delta_y <- y - y_ref
summed_decomposition <- apply(get_result(deeplift), c(1, 3), FUN = sum) # dim 2 is the input feature dim

# Show the mean squared error
mean((delta_y - summed_decomposition)^2)

## ---- eval = FALSE----------------------------------------------------------------------
#  # The global variant (argument 'data' is no longer required)
#  cw_global <- ConnectionWeights$new(converter,
#    times_input = FALSE,
#    ... # other arguments inherited from 'InterpretingMethod'
#  )
#  
#  # The local variant (argument 'data' is required)
#  cw_local <- ConnectionWeights$new(converter, data,
#    times_input = TRUE,
#    ... # other arguments inherited from 'InterpretingMethod'
#  )

## ---------------------------------------------------------------------------------------
# Apply global Connection Weights method
cw_global <- ConnectionWeights$new(converter, times_input = FALSE)

# Show the result
get_result(cw_global)

## ---------------------------------------------------------------------------------------
# Create data
data <- array(c(0.1, 0.4, 0.6), dim = c(3, 1))

# Apply local Connection Weights method
cw_local <- ConnectionWeights$new(converter, data, times_input = TRUE)

# Show the result
get_result(cw_local)

## ---------------------------------------------------------------------------------------
library(torch)
library(neuralnet)
set.seed(45)

# Model for tabular data
# We use the iris dataset for tabular data
tab_data <- as.matrix(iris[, -5])
tab_data <- t((t(tab_data) - colMeans(tab_data)) / rowMeans((t(tab_data) - colMeans(tab_data))^2))
tab_names <- colnames(tab_data)
out_names <- unique(iris$Species)

tab_model <- neuralnet(Species ~ .,
  data = data.frame(tab_data, Species = iris$Species),
  linear.output = FALSE,
  hidden = 10
)

# Model for image data
img_data <- array(rnorm(5 * 32 * 32 * 3), dim = c(5, 3, 32, 32))

img_model <- nn_sequential(
  nn_conv2d(3, 5, c(3, 3)),
  nn_relu(),
  nn_avg_pool2d(c(2, 2)),
  nn_conv2d(5, 10, c(2, 2)),
  nn_relu(),
  nn_avg_pool2d(c(2, 2)),
  nn_flatten(),
  nn_linear(490, 3),
  nn_softmax(2)
)

# Create converter
tab_conv <- Converter$new(tab_model,
  input_dim = c(4),
  input_names = tab_names,
  output_names = out_names
)

img_conv <- Converter$new(img_model, input_dim = c(3, 32, 32))

# Apply Gradient x Input
tab_grad <- Gradient$new(tab_conv, tab_data, times_input = TRUE)
img_grad <- Gradient$new(img_conv, img_data, times_input = TRUE)

## ---- eval = FALSE----------------------------------------------------------------------
#  # You can use the class method
#  method$get_result(type = "array")
#  # or you can use the S3 method
#  get_result(method, type = "array")

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
# Apply method 'Gradient x Input' for classes 1 ('setosa')  and 3 ('virginica')
tab_grad <- Gradient$new(tab_conv, tab_data,
  output_idx = c(1, 3),
  times_input = TRUE
)
# Get result
result_array <- tab_grad$get_result()
# You can also use the S3 function 'get_result'
result_array <- get_result(tab_grad)

# Show the result for datapoint 1 and 10
result_array[c(1, 10), , ]

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
# Apply method 'Gradient' for outputs 1  and 2
img_grad <- Gradient$new(img_conv, img_data, output_idx = c(1, 2))
# Get result
result_array <- img_grad$get_result()
# You can also use the S3 function 'get_result'
result_array <- get_result(img_grad)

# Show the result
str(result_array)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
library(keras)

first_input <- layer_input(shape = c(10, 10, 2))
second_input <- layer_input(shape = c(11))
tmp <- first_input %>%
  layer_conv_2d(2, c(2, 2), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 11)
output <- layer_add(c(tmp, second_input)) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")
model <- keras_model(
  inputs = c(first_input, second_input),
  outputs = output
)

conv <- Converter$new(model)
data <- lapply(
  list(c(10, 10, 2), c(11)),
  function(x) array(rnorm(5 * prod(x)), dim = c(5, x))
)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
# Apply method 'Gradient' for outputs 1  and 2
grad <- Gradient$new(conv, data, output_idx = c(1, 2), channels_first = FALSE)
# Get result
result_array <- grad$get_result()
# You can also use the S3 function 'get_result'
result_array <- get_result(grad)

# Show the result
str(result_array)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
library(keras)

first_input <- layer_input(shape = c(10, 10, 2))
second_input <- layer_input(shape = c(11))
tmp <- first_input %>%
  layer_conv_2d(2, c(2, 2), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 11)
first_output <- layer_add(c(tmp, second_input)) %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")
second_output <- layer_concatenate(c(tmp, second_input)) %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")
model <- keras_model(
  inputs = c(first_input, second_input),
  outputs = c(first_output, second_output)
)

conv <- Converter$new(model)
data <- lapply(
  list(c(10, 10, 2), c(11)),
  function(x) array(rnorm(5 * prod(x)), dim = c(5, x))
)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()------------------
# Apply method 'Gradient' for outputs 1 and 2 in the first and
# for outputs 1 and 3 in the second output layer
grad <- Gradient$new(conv, data,
  output_idx = list(c(1, 2), c(1, 3)),
  channels_first = FALSE
)
# Get result
result_array <- grad$get_result()
# You can also use the S3 function 'get_result'
result_array <- get_result(grad)

# Show the result
str(result_array)

## ---- echo = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
options(width = 500)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
head(get_result(tab_grad, "data.frame"), 5)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
head(get_result(img_grad, "data.frame"), 5)

## ---- fig.width= 8, fig.height=6, eval = torch::torch_is_installed() & keras::is_keras_available()----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library(ggplot2)
library(neuralnet)

# get the result from the tabular model
df <- get_result(tab_grad, "data.frame")

# calculate mean absolute gradient
df <- aggregate(df$value,
  by = list(feature = df$feature, class = df$output_node),
  FUN = function(x) mean(abs(x))
)

ggplot(df) +
  geom_bar(aes(x = feature, y = x, fill = class),
    stat = "identity",
    position = "dodge"
  ) +
  ggtitle("Mean over absolut values of the gradients") +
  xlab("Input feature") +
  ylab("Mean(abs(gradients))") +
  theme_bw()

## ---- eval = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Class method
#  method$plot(
#    data_idx = 1,
#    output_idx = NULL,
#    aggr_channels = "sum",
#    as_plotly = FALSE,
#    same_scale = FALSE
#  )
#  
#  # or the S3 method
#  plot(method,
#    data_idx = 1,
#    output_idx = NULL,
#    aggr_channels = "sum",
#    as_plotly = FALSE,
#    same_scale = FALSE
#  )

## ---- fig.width = 8, fig.height=5, eval = torch::torch_is_installed() & keras::is_keras_available()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot for output classes '1' (setosa) and '3' (virginica) and
# data points '1' and '70'
p <- plot(tab_grad, output_idx = c(1, 3), data_idx = c(1, 70))

# Although it's not a ggplot2 object ...
class(p)

# ... it can be treated as one
p +
  ggplot2::theme_bw() +
  ggplot2::ggtitle("My first 'innsight'-plot")

## ---- fig.width = 8, fig.height=3, eval = torch::torch_is_installed() & keras::is_keras_available()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# In addition, you can use all the options of the class 'innsight_ggplot2',
# e.g. getting the corresponding ggplot2 object
class(p[[1, 1]])

# or creating a subplot
p[2, 1:2]

## ---- fig.width = 8, fig.height=4, echo = TRUE, eval = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # You can do the same with the plotly-based plots
#  p <- plot(tab_grad, output_idx = c(1, 3), data_idx = c(1, 70), as_plotly = TRUE)
#  
#  # Show plot (it also includes a drop down menu for selecting the colorscale)
#  p

## ---- fig.width = 8, fig.height=4, echo = FALSE, message=FALSE, eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # You can do the same with the plotly-based plots
#  p <- plot(tab_grad, output_idx = c(1, 3), data_idx = c(1, 70), as_plotly = TRUE)
#  
#  # Show plot (it also includes a drop down menu for selecting the colorscale)
#  plotly::config(print(p))

## ---- fig.width = 8, fig.height=5, eval = torch::torch_is_installed() & keras::is_keras_available()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# We can do the same for models with image data. In addition, you can define
# the aggregation function for the channels
p <- plot(img_grad,
  output_idx = c(1, 2), data_idx = c(1, 4),
  aggr_channels = "norm"
)

# Although it's not a ggplot2 object ...
class(p)

# ... it can be treated as one
p +
  ggplot2::theme_bw() +
  ggplot2::scale_fill_viridis_c() +
  ggplot2::ggtitle("My first 'innsight'-plot")

## ---- fig.width = 8, fig.height=5, eval = torch::torch_is_installed() & keras::is_keras_available()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# You can also do custom modifications of the results, e.g.
# taking the absolute value of all results. But the
# shape has to be the same after the modification!
result <- tab_grad$result

# The model has only one input (inner list) and one output layer (outer list), so
# we need to modify only a single entry
str(result)

# Take the absolute value and save it back to the object 'img_grad'
tab_grad$result[[1]][[1]] <- abs(result[[1]][[1]])

# Show the result
plot(tab_grad, output_idx = c(1, 3), data_idx = c(1, 70))

## ---- eval = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Class method
#  method$boxplot(
#    output_idx = NULL,
#    data_idx = "all",
#    ref_data_idx = NULL,
#    aggr_channels = "sum",
#    preprocess_FUN = abs,
#    as_plotly = FALSE,
#    individual_data_idx = NULL,
#    individual_max = 20
#  )
#  
#  # or the S3 method
#  boxplot(method,
#    output_idx = NULL,
#    data_idx = "all",
#    ref_data_idx = NULL,
#    aggr_channels = "sum",
#    preprocess_FUN = abs,
#    as_plotly = FALSE,
#    individual_data_idx = NULL,
#    individual_max = 20
#  )

## ---- fig.width = 8, fig.height=5, eval = torch::torch_is_installed() & keras::is_keras_available()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a boxplot for output classes '1' (setosa) and '3' (virginica)
p <- boxplot(tab_grad, output_idx = c(1, 3))

# Although, it's not a ggplot2 object ...
class(p)

# ... it can be treated as one
p +
  ggplot2::theme_bw() +
  ggplot2::ggtitle("My first 'innsight'-boxplot!")
# You can also select only the indices of the class 'setosa'
# and add a reference data point of another class ('versicolor')
boxplot(tab_grad, output_idx = c(1, 3), data_idx = 1:50, ref_data_idx = c(60))

## ---- fig.width = 8, fig.height=4, echo = TRUE, eval = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # You can do the same with the plotly-based plots
#  p <- boxplot(tab_grad,
#    output_idx = c(1, 3), data_idx = 1:50,
#    ref_data_idx = 60, as_plotly = TRUE
#  )
#  
#  # Show plot (it also includes a drop down menu for selecting the reference data
#  # point and toggle the plot type 'Boxplot' or 'Violin')
#  p

## ---- fig.width = 8, fig.height=4, echo = FALSE, message=FALSE, eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # You can do the same with the plotly-based plots
#  p <- boxplot(tab_grad,
#    output_idx = c(1, 3), data_idx = 1:50,
#    ref_data_idx = 60, as_plotly = TRUE
#  )
#  
#  # Show plot (it also includes a drop down menu for selecting the reference data
#  # point and toggle the plot type Boxplot or Violin)
#  plotly::config(print(p))

## ---- fig.width=8, fig.height=4, eval = torch::torch_is_installed() & keras::is_keras_available()-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# We can do the same for models with image data. In addition, you can define
# the aggregation function for the channels
p <- boxplot(img_grad, output_idx = c(1, 2), aggr_channels = "norm")

# Although it's not a ggplot2 object ...
class(p)

# ... it can be treated as one
p +
  ggplot2::theme_bw() +
  ggplot2::coord_flip() +
  ggplot2::ggtitle("My first 'innsight'-boxplot")

## ---- fig.width = 8, fig.height=4, echo = TRUE, eval = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # You can do the same with the plotly-based plots
#  p <- boxplot(img_grad,
#    output_idx = c(1, 2), aggr_channels = "norm",
#    as_plotly = TRUE
#  )
#  
#  # Show plot (it also includes a drop down menu for selecting the colorscale,
#  # another menu for toggling between the plot types 'Heatmap' and 'Contour'
#  # and a scale for selecting the respective percentile)
#  p

## ---- fig.width = 8, fig.height=4, echo = FALSE, message=FALSE, eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # You can do the same with the plotly-based plots
#  p <- boxplot(img_grad,
#    output_idx = c(1, 2), aggr_channels = "norm",
#    as_plotly = TRUE
#  )
#  plotly::config(print(p))

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library(keras)
library(torch)

# Create model with tabular data as inputs and one output layer
model <- keras_model_sequential() %>%
  layer_dense(50, activation = "relu", input_shape = c(5)) %>%
  layer_dense(20, activation = "relu") %>%
  layer_dense(3, activation = "softmax")

converter <- Converter$new(model)

data <- array(rnorm(5 * 50), dim = c(50, 5))
res_simple <- Gradient$new(converter, data)

# Create model with images as inputs and two output layers
input_image <- layer_input(shape = c(10, 10, 3))
conv_part <- input_image %>%
  layer_conv_2d(5, c(2, 2), activation = "relu", padding = "same") %>%
  layer_average_pooling_2d() %>%
  layer_conv_2d(4, c(2, 2)) %>%
  layer_activation(activation = "softplus") %>%
  layer_flatten()

output_1 <- conv_part %>%
  layer_dense(50, activation = "relu") %>%
  layer_dense(3, activation = "softmax")

output_2 <- conv_part %>%
  layer_dense(50, activation = "relu") %>%
  layer_dense(3, activation = "softmax")

keras_model_concat <- keras_model(
  inputs = input_image,
  outputs = c(output_1, output_2)
)

converter <- Converter$new(keras_model_concat)

data <- array(rnorm(10 * 10 * 3 * 5), dim = c(5, 10, 10, 3))
res_one_input <- Gradient$new(converter, data,
  channels_first = FALSE,
  output_idx = list(1:3, 1:3)
)

# Create model with images and tabular data as inputs and two
# output layers
input_image <- layer_input(shape = c(10, 10, 3))
input_tab <- layer_input(shape = c(10))

conv_part <- input_image %>%
  layer_conv_2d(5, c(2, 2), activation = "relu", padding = "same") %>%
  layer_average_pooling_2d() %>%
  layer_conv_2d(4, c(2, 2)) %>%
  layer_activation(activation = "softplus") %>%
  layer_flatten()

output_1 <- layer_concatenate(list(conv_part, input_tab)) %>%
  layer_dense(50, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(3, activation = "softmax")

output_2 <- layer_concatenate(list(conv_part, input_tab)) %>%
  layer_dense(3, activation = "softmax")

keras_model_concat <- keras_model(
  inputs = list(input_image, input_tab),
  outputs = list(output_1, output_2)
)

converter <- Converter$new(keras_model_concat)

data <- lapply(list(c(10, 10, 3), c(10)), function(x) torch_randn(c(5, x)))
res_two_inputs <- Gradient$new(converter, data,
  times_input = TRUE,
  channels_first = FALSE,
  output_idx = list(1:3, 1:3)
)

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot for output node 1 and 2 in the first output layer and
# data points 1 and 3
p <- plot(res_one_input, output_idx = c(1, 2), data_idx = c(1, 3))

# It's not an ggplot2 object
class(p)

# The slot 'grobs' only contains a single entry
p@grobs

# It's not a multiplot
p@multiplot

# Therefore, slots 'output_strips' and 'col_dims' are empty lists
p@output_strips
p@col_dims

## ---- fig.height=6, fig.width=8, eval = torch::torch_is_installed() & keras::is_keras_available()-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_one_input, output_idx = c(1, 2), data_idx = c(1, 3))

# Now we can add geoms, themes and scales as usual for ggplot2 objects
df <- data.frame(
  x = c(0.5, 0.5, 10.5, 10.5, 0.5),
  y = c(0.5, 10.5, 10.5, 0.5, 0.5)
)
new_p <- p +
  geom_path(df, mapping = aes(x = x, y = y), color = "red", size = 3) +
  theme_bw() +
  scale_fill_viridis_c()

# This object is still an 'innsight_ggplot2' object...
class(new_p)

# ... but all ggplot2 geoms, themes and scales are added
new_p
# If the respective plot allows it, you can also use the already existing
# mapping function and data:
boxplot(res_simple, output_idx = 1:2) +
  geom_jitter(width = 0.3, alpha = 0.4)

## ---- fig.height=6, fig.width=8, eval = torch::torch_is_installed() & keras::is_keras_available()-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_one_input, output_idx = c(1, 2), data_idx = c(1, 3))

# Show the whole plot
p
# Now you can select specific rows and columns for in-depth investigation,
# e.g. only the result for output "Y1"
p_new <- p[1:2, 1]

# It's still an obeject of class 'innsight_ggplot2'
class(p_new)

# Show the subplot
p_new

## ---- fig.height=6, fig.width=8, eval = torch::torch_is_installed() & keras::is_keras_available()-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_one_input, output_idx = c(1, 2), data_idx = c(1, 3))

# Show the whole plot
p
# Now you can select a single plot by passing the row and column index,
# e.g. the plot for output "Y1" and data point 3
p_new <- p[[2, 1]]

# This time a ggplot2 object is returned
class(p_new)

# Show the new plot
p_new

## ---- fig.height=3, fig.width=8, eval = torch::torch_is_installed() & keras::is_keras_available()-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_one_input, output_idx = c(1, 2), data_idx = 1)

# All methods behave the same and return a ggplot2 object
class(print(p))
class(show(p))
class(plot(p))

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a plot for output node 1 in the first output layer and node 2 in the
# second output layer and data points 1 and 3
p <- plot(res_two_inputs, output_idx = list(1, c(1, 2)), data_idx = c(1, 3))

# It's not a ggplot2 object
class(p)

# In this case, 'grobs' is a 2x6 matrix
p@grobs

# It's a multiplot
p@multiplot

# Slot 'output_strips' is a list with the three labels for the output nodes
# and the theme for the strips
str(p@output_strips, max.level = 1)

# Slot 'col_dims' contains the number of columns for each output node
p@col_dims

## ---- fig.height=4, fig.width=10, eval = torch::torch_is_installed() & keras::is_keras_available()----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_two_inputs, output_idx = list(1, 2), data_idx = c(1,2))

# Show the whole plot
p
# Select a restyled subplot (default)
p[1, 1:2]
# The same plot as shown in the whole plot
p[1, 1:2, restyle = FALSE]

## ---- fig.height=4, fig.width=10, eval = torch::torch_is_installed() & keras::is_keras_available()----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_two_inputs, output_idx = list(1, 2), data_idx = 1)

# All methods behave the same and return a ggplot2 object
class(print(p))
class(show(p))
class(plot(p))

# You can also pass additional arguments to the method 'arrangeGrob',
# e.g. double the width of both images
print(p, widths = c(2, 1, 2, 1))

## ---- fig.height=7, fig.width=12, eval = torch::torch_is_installed() & keras::is_keras_available()----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create plot
p <- plot(res_two_inputs, output_idx = list(1, 2), data_idx = c(1, 3))

# Remove colorbar in the plot for data point 3 and output 'Y1' in output
# layer 1 (in such situations the `restyle` argument is useful)
p[2, 1] <- p[2, 1, restyle = FALSE] + guides(fill = "none")

# Change colorscale in the plot for data point 1 and output 'Y2' in output
# layer 2
p[1, 3:4] <- p[1, 3:4, restyle = FALSE] + scale_fill_gradient2(limit = c(-1, 1))

# Change the theme in all plots for data point 3
p[2, ] <- p[2, , restyle = FALSE] + theme_dark()

# Show the result with all changes
p

## ---- eval = torch::torch_is_installed() & keras::is_keras_available()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a plot for output node 1 in the first layer and output node 2 in the
# second layer and data point 1 and 3
p <- plot(res_two_inputs,
  output_idx = list(1, 2), data_idx = c(1, 3),
  as_plotly = TRUE
)

# Slot 'plots' is a 2x4 matrix (2 data points, 2 output nodes and 2 input layers)
p@plots

# Slot 'shapes' contains two 2x4 matrices with the corresponding shape objects
p@shapes

# The same for the annotations
p@annotations

# The model has multiple input layers, so the slot 'multiplot' is TRUE
p@multiplot

# The overall layout is stored in the slot 'layout'
str(p@layout, max.level = 1)

# 'col_dims' assigns the label of the additional strips to the respective column
p@col_dims

## ---- eval=FALSE, echo = TRUE-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Create plot
#  p <- plot(res_two_inputs,
#    output_idx = list(1, 2), data_idx = c(1, 3),
#    as_plotly = TRUE
#  )
#  
#  # Show the whole plot
#  p

## ---- eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available(), echo = FALSE, fig.height=4,out.width= "100%", fig.width=20------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Create plot
#  p <- plot(res_two_inputs,
#    output_idx = list(1, 2), data_idx = c(1, 3),
#    as_plotly = TRUE
#  )
#  
#  # Show the whole plot
#  plotly::config(print(p))

## ---- eval=FALSE, echo = TRUE-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Now you can select specific rows and columns for in-depth investigation,
#  # e.g. only the result for output "Y2"
#  p_new <- p[1:2, 3:4]
#  
#  # It's still an object of class 'innsight_plotly'
#  class(p_new)
#  #> [1] "innsight_plotly"
#  #> attr(,"package")
#  #> [1] "innsight"
#  
#  # Show the subplot
#  p_new

## ---- eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available(), echo = FALSE, fig.height=5, fig.width=10------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Now you can select specific rows and columns for in-depth investigation,
#  # e.g. only the result for output "Y2"
#  p_new <- p[1:2, 3:4]
#  
#  # Show the subplot
#  plotly::config(print(p_new))

## ---- eval=FALSE, echo = TRUE-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Create plot
#  p <- plot(res_two_inputs,
#    output_idx = list(1, 2), data_idx = c(1, 3),
#    as_plotly = TRUE
#  )
#  
#  # Now you can select a single plot by passing the row and column index,
#  # e.g. the plot for output "Y1", data point 3 and the second input layer
#  p_new <- p[[2, 2]]
#  
#  # It's a plotly object
#  class(p_new)
#  #> [1] "plotly"     "htmlwidget"
#  
#  # Show the plot
#  p_new

## ---- eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available(), echo = FALSE, fig.height=5, fig.width=10------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Create plot
#  p <- plot(res_two_inputs,
#    output_idx = list(1, 2), data_idx = c(1, 3),
#    as_plotly = TRUE
#  )
#  
#  # Now you can select a single plot by passing the row and column index,
#  # e.g. the plot for output "Y1", data point 3 and the second input layer
#  p_new <- p[[2, 2]]
#  
#  # Show the subplot
#  p_new

## ---- fig.height=4, fig.width=10, eval = FALSE--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Create plot
#  p <- plot(res_two_inputs,
#    output_idx = list(1, 2), data_idx = 1,
#    as_plotly = TRUE
#  )
#  
#  # All methods behave the same and return a plotly object
#  class(print(p))
#  #> [1] "plotly"     "htmlwidget"
#  class(show(p))
#  #> [1] "plotly"     "htmlwidget"
#  class(plot(p))
#  #> [1] "plotly"     "htmlwidget"
#  #>
#  # You can also pass additional arguments to the method 'plotly::subplot',
#  # e.g. the margins
#  print(p, margin = c(0.03, 0.03, 0.01, 0.01))

## ---- fig.height=4, fig.width=10, echo = FALSE, eval=Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed() & keras::is_keras_available()------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  # Create plot
#  p <- plot(res_two_inputs,
#    output_idx = list(1, 2), data_idx = 1,
#    as_plotly = TRUE
#  )
#  
#  # You can also pass additional arguments to the method 'plotly::subplot',
#  # e.g. the margins
#  plotly::config(print(p,
#    margin = c(0.03, 0.03, 0.01, 0.01),
#    widths = c(0.35, 0.15, 0.15, 0.35)
#  ))

