## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- eval = FALSE------------------------------------------------------------
#  model <- list()
#  model$input_dim <- c(4)
#  model$input_names <- list(c("Feat1", "Feat2", "Feat3", "Feat4")) # optional
#  model$output_dim <- c(2) # optional
#  model$output_names <- list(c("Out1", "Out2")) # optional
#  model$layers <- list() # see next section

## ---- eval = FALSE------------------------------------------------------------
#  # Define dense layer
#  dense_layer <- list(
#    type = 'Dense',
#    weight = matrix(rnorm(5 * 2), 2, 5),
#    bias = rnorm(2),
#    activation_name = 'tanh',
#    dim_in = 5, # optional
#    dim_out = 2 # optional
#  )
#  
#  # Adding layer to model
#  model$layers$custom_dense <- dense_layer

## ---- eval = FALSE------------------------------------------------------------
#  conv_1D <- list(
#        type = "Conv1D",
#        weight = array(rnorm(8*3*2), dim = c(8,3,2)),
#        bias = rnorm(8),
#        activation_name = "tanh",
#        dim_in = c(3, 10), # optional
#        dim_out = c(8, 9) # optional
#      )
#  # Adding layer to model
#  model$layers$custom_conv1D <- conv_1D

## ---- eval = FALSE------------------------------------------------------------
#  conv_2D <- list(
#        type = "Conv2D",
#        weight = array(rnorm(8*3*2*4), dim = c(8,3,2,4)),
#        bias = rnorm(8),
#        padding = c(1,1,0,0),
#        dilation = c(1,2),
#        activation_name = "relu",
#        dim_in = c(3, 10, 10) # optional
#      )
#  # Adding layer to model
#  model$layers$custom_conv2D <- conv_2D

## ---- eval = FALSE------------------------------------------------------------
#  avg_pool2D <- list(
#    type = "AveragePooling2D",
#    kernel_size = c(2,2)
#    )
#  # Adding layer to model
#  model$layers$custom_avgpool2D <- avg_pool2D

## ---- eval = FALSE------------------------------------------------------------
#  flatten <- list(
#    type = "Flatten"
#    )
#  # Adding layer to model
#  model$layers$custom_flatten <- flatten

## ---- eval = torch::torch_is_installed()--------------------------------------
library(innsight)

model <- list()
model$input_dim <- 5
model$input_names <- list(c("Feat1", "Feat2", "Feat3", "Feat4", "Feat5"))
model$output_dim <- 2
model$output_names <- list(c("Cat", "no-Cat"))
model$layers$Layer_1 <-
  list(
    type = "Dense",
    weight = matrix(rnorm(5 * 20), 20, 5),
    bias = rnorm(20),
    activation_name = "tanh",
    dim_in = 5L,
    dim_out = 20L
  )
model$layers$Layer_2 <-
  list(
    type = "Dense",
    weight = matrix(rnorm(20 * 2), 2, 20),
    bias = rnorm(2),
    activation_name = "softmax",
    dim_in = 20L,
    dim_out = 2L
  )

# Convert the model
converter <- Converter$new(model)

## ---- eval = torch::torch_is_installed()--------------------------------------
library(innsight)

model <- list()
model$input_dim <- c(3, 30, 30)
model$output_dim <- c(2)

model$layers$Layer_1 <- list(
  type = "Conv2D",
  weight = array(rnorm(10*3*5*5), dim = c(10,3,5,5)),
  bias = rnorm(10),
  activation_name = "relu"
)
model$layers$Layer_2 <- list(
  type = 'AveragePooling2D',
  kernel_size = c(2,2),
  dim_in = c(10, 26, 26) # optional
)
model$layers$Layer_3 <- list(
  type = 'Conv2D',
  weight = array(rnorm(8*10*4*4), dim = c(8,10,4,4)),
  bias = rnorm(8),
  activation_name = "relu",
  padding = c(2,2,3,3),
  dim_out = c(8, 16, 14) # optional
)
model$layers$Layer_4 <- list(
  type = "AveragePooling2D",
  kernel_size = c(2,2),
  strides = c(2,3)
)
model$layers$Layer_5 <- list(
  type = "Flatten",
  dim_out = c(320) # optional
)
model$layers$Layer_6 <- list(
  type = "Dense",
  weight = array(rnorm(320*2), dim = c(2, 320)),
  bias = rnorm(2),
  activation_name = "softmax"
)

# Convert the model
converter <- Converter$new(model)

