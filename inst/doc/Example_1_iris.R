## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  size = "huge",
  collapse = TRUE,
  comment = "#>",
  eval = torch::torch_is_installed(),
  fig.align = "center",
  out.width = "95%"
)

## ---- echo = FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sys.setenv(LANG = "en_US.UTF-8")
set.seed(1111)
options(width = 500)
torch::torch_manual_seed(1111)

## ----example_1_train, echo = TRUE---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library(innsight)
library(torch)

# Set seeds for reproducibility
set.seed(42)
torch_manual_seed(42)

# Prepare data
data(iris)
x <- torch_tensor(as.matrix(iris[, -5]))
# normalize input to [-1, 1]
x <- 2 * (x - x$min(1)[[1]]) / (x$max(1)[[1]] - x$min(1)[[1]]) - 1
y <- torch_tensor(as.integer(iris[, 5]))

# Define model (`torch::nn_sequential`)
model <- nn_sequential(
  nn_linear(4, 30),
  nn_relu(),
  nn_dropout(0.3),
  nn_linear(30, 10),
  nn_relu(),
  nn_linear(10, 3),
  nn_softmax(dim = 2)
)

# Train model
optimizer <- optim_adam(model$parameters, lr = 0.001)
for (t in 1:2500) {
  y_pred <- torch_log(model(x))
  loss <- nnf_nll_loss(y_pred, y)
  if (t %% 250 == 0) {
    cat("Loss: ", as.numeric(loss), "\n")
  }
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
}

## ----example_1_conv_1---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create the converter object
converter <- convert(model, input_dim = c(4))

## ----example_1_conv_2, eval = torch::torch_is_installed()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create `Converter` object (with custom labels)
converter <- convert(model,
  input_dim = c(4),
  input_names = c("Sepal (length)", "Sepal (width)", "Petal (length)", "Petal (width)"),
  output_names = c("Setosa", "Versicolor", "Virginica")
)

## ---- echo=FALSE--------------------------------------------------------------
options(width = 80)

## -----------------------------------------------------------------------------
converter

## -----------------------------------------------------------------------------
grad_no_softmax <- run_grad(converter, x, ignore_last_act = TRUE)

## ---- message = FALSE, results = 'hide'---------------------------------------
grad_softmax <- run_grad(converter, x, ignore_last_act = FALSE)

## ---- message = FALSE, results = 'hide'---------------------------------------
lrp_eps <- run_lrp(converter, x, rule_name = "epsilon", rule_param = 0.01)

## ---- message = FALSE, results = 'hide'---------------------------------------
x_ref <- x$mean(1, keepdim = TRUE) # ref value needs the shape (1,4)
deeplift_mean <- run_deeplift(converter, x, x_ref = x_ref)

## -----------------------------------------------------------------------------
deeplift_mean

## ---- echo=FALSE------------------------------------------------------------------------
options(width = 90)

## ---------------------------------------------------------------------------------------
# Get result as a `data.frame` using the class method
head(grad_no_softmax$get_result(type = "data.frame"), 5)

# Get result as `array` (default) using the generic S3 function
str(get_result(grad_no_softmax))

## ---- fig.height=6, fig.keep='all', fig.width=9-----------------------------------------
# Show data point 1 and 111 for output node 1 (Setosa) and 2 (Versicolor)
plot(grad_no_softmax, data_idx = c(1, 111), output_idx = c(1, 2)) +
  ggplot2::theme_bw()

## ---- fig.height=4, fig.keep='all', fig.width=9, eval = FALSE---------------------------
#  # Show data point 1 for output node 1 (Setosa) and 2 (Versicolor)
#  plot(deeplift_mean, data_idx = 1, output_idx = c(1, 2), as_plotly = TRUE)

## ---- fig.height=4, echo = FALSE, fig.width=9, message = FALSE, eval = Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed()----
#  # Show data point 1 for output node 1 (Setosa) and 2 (Versicolor)
#  p <- plot(deeplift_mean, data_idx = 1, output_idx = c(1, 2), as_plotly = TRUE)
#  plotly::config(print(p, shareY = TRUE))

## ---- fig.height=6, fig.keep='all', fig.width=9-----------------------------------------
# Summarized results for output node 1 (Setosa) and 2 (Versicolor) and
# reference value of index 3
boxplot(grad_no_softmax, output_idx = c(1, 2), ref_data_idx = 3, preprocess_FUN = abs) +
  ggplot2::theme_bw()

## ---- fig.height=4, fig.keep='all', fig.width=9, eval = FALSE---------------------------
#  # Show boxplot only for instances of class setosa for output node 1 (Setosa)
#  # and 2 (Versicolor)
#  boxplot(lrp_eps, data_idx = 1:50, output_idx = c(1, 2), as_plotly = TRUE)

## ---- fig.height=4, echo = FALSE, fig.width=9, message = FALSE, eval = Sys.getenv("RENDER_PLOTLY", unset = 0) == 1 & torch::torch_is_installed()----
#  # Show data point 1 for output node 1 (Setosa) and 2 (Versicolor)
#  p <- boxplot(lrp_eps, data_idx = 1:50, output_idx = c(1, 2), as_plotly = TRUE)
#  plotly::config(print(p, shareY = TRUE))

