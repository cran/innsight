## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 4,
  fig.align = "center"
)

set.seed(999)

## ---- eval = FALSE------------------------------------------------------------
#  # Step 1: Model creation and converting
#  model = ...
#  converter <- Converter$new(model)
#  
#  # Step 2: Apply selected method to your data
#  result <- Method$new(converter, data)
#  
#  # Step 3: Plot the results
#  plot(result)
#  boxplot(result)

## ---- eval = torch::torch_is_installed()--------------------------------------
library(torch)
library(innsight)
torch_manual_seed(123)

# Create model
model <- nn_sequential(
  nn_linear(3, 10),
  nn_relu(),
  nn_linear(10, 2, bias = FALSE),
  nn_softmax(2)
)
# Convert the model
conv_dense <- Converter$new(model, input_dim = c(3))
# Convert model with input and output names
conv_dense_with_names <- 
  Converter$new(model, input_dim = c(3),
                input_names = list(c("Price", "Weight", "Height")),
                output_names = list(c("Buy it!", "Don't buy it!")))
# Show output names
conv_dense_with_names$model_dict$output_names

## ---- eval = torch::torch_is_installed()--------------------------------------
nn_flatten <- nn_module(
    classname = "nn_flatten",
    initialize = function(start_dim = 2, end_dim = -1) {
      self$start_dim <- start_dim
      self$end_dim <- end_dim
    },
    forward = function(x) {
      torch_flatten(x, start_dim = self$start_dim, end_dim = self$end_dim)
    }
  )

## ---- eval = torch::torch_is_installed()--------------------------------------
# Create CNN for images of size (3, 28, 28)
model <- nn_sequential(
  nn_conv2d(3, 5, c(2, 2)),
  nn_relu(),
  nn_max_pool2d(c(1,2)),
  nn_conv2d(5, 6, c(2, 3), stride = c(1, 2)),
  nn_relu(),
  nn_conv2d(6, 2, c(2, 2), dilation = c(1, 2), padding = c(5,4)),
  nn_relu(),
  nn_avg_pool2d(c(2,2)),
  nn_flatten(),
  nn_linear(48, 5),
  nn_softmax(2)
)

# Convert the model
conv_cnn <- Converter$new(model, input_dim = c(3, 10, 10))

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
library(keras)
tensorflow::set_random_seed(42)

# Create model
model <- keras_model_sequential()
model <- model %>%
  layer_dense(10, input_shape = c(5), activation = "softplus") %>%
  layer_dense(8, use_bias = FALSE, activation = "tanh") %>%
  layer_dropout(0.2) %>%
  layer_dense(4, activation = "softmax")

# Convert the model
conv_dense <- Converter$new(model)

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
library(keras)

# Create model
model <- keras_model_sequential()
model <- model %>%
  layer_conv_2d(4, c(5,4), input_shape = c(10,10,3), activation = "softplus") %>%
  layer_max_pooling_2d(c(2,2), strides = c(1,1)) %>%
  layer_conv_2d(6, c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_conv_2d(4, c(2,2), strides = c(2,1), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(5, activation = "softmax")

# Convert the model
conv_cnn <- Converter$new(model)

## ---- eval = torch::torch_is_installed()--------------------------------------
library(neuralnet)
data(iris)

# Create model
model <- neuralnet(Species ~ Petal.Length + Petal.Width, iris, 
                   linear.output = FALSE)

# Convert model
conv_dense <- Converter$new(model)
# Show input names
conv_dense$model_dict$input_names
# Show output names
conv_dense$model_dict$output_names

## ---- results='hide', message=FALSE, eval = keras::is_keras_available() & torch::torch_is_installed()----
# Apply method 'Gradient' for the dense network
grad_dense <- Gradient$new(conv_dense, iris[-c(1,2,5)])

# Apply method 'Gradient x Input' for CNN
x <- torch_randn(c(10,3,10,10))
grad_cnn <- Gradient$new(conv_cnn, x, times_input = TRUE)

## ---- results='hide', message=FALSE, eval = keras::is_keras_available() & torch::torch_is_installed()----
# Apply method 'SmoothGrad' for the dense network
smooth_dense <- SmoothGrad$new(conv_dense, iris[-c(1,2,5)])

# Apply method 'SmoothGrad x Input' for CNN
x <- torch_randn(c(10,3,10,10))
smooth_cnn <- SmoothGrad$new(conv_cnn, x, times_input = TRUE)

## ---- results='hide', message=FALSE, eval = keras::is_keras_available() & torch::torch_is_installed()----
# Apply method 'LRP' for the dense network
lrp_dense <- LRP$new(conv_dense, iris[-c(1,2,5)])

# Apply method 'LRP' for CNN with alpha-beta-rule
x <- torch_randn(c(10,10,10,3))
lrp_cnn <- LRP$new(conv_cnn, x, rule_name = "alpha_beta", rule_param = 1,
                   channels_first = FALSE)

## ---- results='hide', message=FALSE, eval = keras::is_keras_available() & torch::torch_is_installed()----
# Define reference value
x_ref <- array(colMeans(iris[-c(1,2,5)]), dim = c(1,2))
# Apply method 'DeepLift' for the dense network
deeplift_dense <- DeepLift$new(conv_dense, iris[-c(1,2,5)], x_ref = x_ref)

# Apply method 'DeepLift' for CNN
x <- torch_randn(c(10,3,10,10))
deeplift_cnn <- DeepLift$new(conv_cnn, x)

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
# Get result (make sure 'grad_dense' is defined!)
result_array <- grad_dense$get_result()

# Show for datapoint 1 and 71 the result
result_array[c(1,71),,]

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
# Get result as data.frame (make sure 'lrp_cnn' is defined!)
result_data.frame <- lrp_cnn$get_result("data.frame")

# Show the first 5 rows
head(result_data.frame, 5)

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
# Get result (make sure 'deeplift_dense' is defined!)
result_torch <- deeplift_dense$get_result("torch_tensor")

# Show for datapoint 1 and 71 the result
result_torch[c(1,71),,]

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
plot(smooth_dense, output_idx = 1:3)
# You can plot several data points at once
plot(smooth_dense, data_idx = c(1,71), output_idx = 1:3)
# Plot result for the first data point and first and fourth output
plot(lrp_cnn, aggr_channels = 'norm', output_idx = c(1,4))

## ---- eval = FALSE------------------------------------------------------------
#  # Create a plotly plot for the first output
#  plot(lrp_cnn, aggr_channels = 'norm', output_idx = c(1), as_plotly = TRUE)

## ---- eval = keras::is_keras_available() & torch::torch_is_installed()--------
boxplot(smooth_dense, output_idx = 1:2)
# Use no preprocess function (default: abs) and plot reference data point
boxplot(smooth_dense, output_idx = 1:3, preprocess_FUN = identity,
        ref_data_idx = c(55))

## ---- eval = FALSE------------------------------------------------------------
#  boxplot(smooth_dense, output_idx = 1:3, preprocess_FUN = identity,
#          ref_data_idx = c(55), as_plotly = TRUE, individual_data_idx = c(1))

