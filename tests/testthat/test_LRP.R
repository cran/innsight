
test_that("LRP: General errors", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_error(LRP$new(model, data))
  expect_error(LRP$new(converter, model))
  expect_error(LRP$new(converter, data, channels_first = NULL))
  expect_error(LRP$new(converter, data, rule_name = "asdf"))
  expect_error(LRP$new(converter, data, rule_param = "asdf"))
  expect_error(LRP$new(converter, data, dtype = NULL))
})


test_that("LRP: Plot and Boxplot", {
  library(neuralnet)
  library(torch)

  data(iris)
  data <- iris[sample.int(150, size = 10), -5]
  nn <- neuralnet(Species ~ .,
                  iris,
                  linear.output = FALSE,
                  hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
  )
  # create an converter for this model
  converter <- Converter$new(nn)

  # Rescale Rule
  lrp <- LRP$new(converter, data, dtype = "double",
  )

  # ggplot2

  # Non-existing data points
  expect_error(plot(lrp, data_idx = c(1,11)))
  expect_error(boxplot(lrp, data_idx = 1:11))
  # Non-existing class
  expect_error(plot(lrp, output_idx = c(5)))
  expect_error(boxplot(lrp, output_idx = c(5)))

  p <- plot(lrp)
  boxp <- boxplot(lrp)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(lrp, data_idx = 1:3)
  boxp <- boxplot(lrp, data_idx = 1:4)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(lrp, data_idx = 1:3, output_idx = 1:3)
  boxp <- boxplot(lrp, data_idx = 1:5, output_idx = 1:3)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # plotly
  library(plotly)

  p <- plot(lrp, as_plotly = TRUE)
  boxp <- boxplot(lrp, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(lrp, data_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(lrp, data_idx = 1:4, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(lrp, data_idx = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(lrp, data_idx = 1:5, output_idx = 1:3, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
})



test_that("LRP: Dense-Net (Neuralnet)", {
  library(neuralnet)
  library(torch)

  data(iris)
  data <- iris[sample.int(150, size = 10), -5]
  nn <- neuralnet(Species ~ .,
                  iris,
                  linear.output = FALSE,
                  hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
  )
  # create an converter for this model
  converter <- Converter$new(nn)

  expect_error(LRP$new(converter, array(rnorm(4 * 2 * 3), dim = c(2, 3, 4))))

  # Simple Rule
  lrp_simple <- LRP$new(converter, data)
  expect_equal(dim(lrp_simple$get_result()), c(10, 4, 3))
  expect_true(
    lrp_simple$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Epsilon Rule
  lrp_eps_default <-
    LRP$new(converter, data, rule_name = "epsilon", dtype = "double")
  expect_equal(dim(lrp_eps_default$get_result()), c(10, 4, 3))
  expect_true(
    lrp_eps_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_eps_1 <- LRP$new(converter, data,
                       rule_name = "epsilon",
                       rule_param = 1,
                       ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_eps_1$get_result()), c(10, 4, 3))
  expect_true(
    lrp_eps_1$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(converter, data,
                            rule_name = "epsilon",
                            dtype = "double",
                            ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_ab_default$get_result()), c(10, 4, 3))
  expect_true(
    lrp_ab_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_ab_2 <- LRP$new(converter, data, rule_name = "epsilon", rule_param = 2)
  expect_equal(dim(lrp_ab_2$get_result()), c(10, 4, 3))
  expect_true(
    lrp_ab_2$get_result(type = "torch.tensor")$dtype == torch_float()
  )
})



test_that("LRP: Dense-Net (keras)", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_error(LRP$new(converter, array(rnorm(4 * 2 * 3), dim = c(2, 3, 4))))

  # Simple Rule
  lrp_simple <- LRP$new(converter, data)
  expect_equal(dim(lrp_simple$get_result()), c(10, 4, 3))
  expect_true(
    lrp_simple$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Epsilon Rule
  lrp_eps_default <-
    LRP$new(converter, data, rule_name = "epsilon", dtype = "double")
  expect_equal(dim(lrp_eps_default$get_result()), c(10, 4, 3))
  expect_true(
    lrp_eps_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_eps_1 <- LRP$new(converter, data,
    rule_name = "epsilon",
    rule_param = 1,
    ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_eps_1$get_result()), c(10, 4, 3))
  expect_true(
    lrp_eps_1$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(converter, data,
    rule_name = "epsilon",
    dtype = "double",
    ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_ab_default$get_result()), c(10, 4, 3))
  expect_true(
    lrp_ab_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_ab_2 <- LRP$new(converter, data, rule_name = "epsilon", rule_param = 2)
  expect_equal(dim(lrp_ab_2$get_result()), c(10, 4, 3))
  expect_true(
    lrp_ab_2$get_result(type = "torch.tensor")$dtype == torch_float()
  )
})

test_that("LRP: Conv1D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 64 * 3), dim = c(4, 64, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(64, 3), kernel_size = 16, filters = 8,
      activation = "softplus"
    ) %>%
    layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
    layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  expect_error(LRP$new(converter, array(rnorm(4 * 2 * 3), dim = c(2, 3, 4))))

  # Simple Rule
  lrp_simple <- LRP$new(converter, data, channels_first = FALSE)
  expect_equal(dim(lrp_simple$get_result()), c(4, 64, 3, 1))
  expect_true(
    lrp_simple$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Epsilon Rule
  lrp_eps_default <- LRP$new(converter, data,
    rule_name = "epsilon",
    dtype = "double", channels_first = FALSE
  )
  expect_equal(dim(lrp_eps_default$get_result()), c(4, 64, 3, 1))
  expect_true(
    lrp_eps_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_eps_1 <- LRP$new(converter, data,
    rule_name = "epsilon",
    rule_param = 1,
    channels_first = FALSE,
    ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_eps_1$get_result()), c(4, 64, 3, 1))
  expect_true(
    lrp_eps_1$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(converter, data,
    rule_name = "epsilon",
    dtype = "double",
    channels_first = FALSE,
    ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_ab_default$get_result()), c(4, 64, 3, 1))
  expect_true(
    lrp_ab_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_ab_2 <- LRP$new(converter, data,
    rule_name = "epsilon",
    rule_param = 2,
    channels_first = FALSE
  )
  expect_equal(dim(lrp_ab_2$get_result()), c(4, 64, 3, 1))
  expect_true(
    lrp_ab_2$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Different rules
  lrp_mixed_rules <- LRP$new(converter, data,
                             rule_name = list(Dense_Layer = "alpha_beta"),
                             rule_param = list(Dense_Layer = 2),
                             channels_first = FALSE)
  expect_equal(dim(lrp_mixed_rules$get_result()), c(4, 64, 3, 1))

  lrp_mixed_rules <- LRP$new(converter, data,
                             rule_name = list(Dense_Layer = "alpha_beta",
                                              Conv1D_Layer = "epsilon"),
                             rule_param = list(Dense_Layer = 2),
                             channels_first = FALSE)
  expect_equal(dim(lrp_mixed_rules$get_result()), c(4, 64, 3, 1))
  expect_error(LRP$new(converter, data,
                       rule_name = list(Flatten = "alpha_beta",
                                        Conv1D_Layer = "epsilon"),
                       rule_param = list(Dense_Layer = 2),
                       channels_first = FALSE))
})

test_that("LRP: Conv2D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 32 * 32 * 3), dim = c(4, 32, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "same"
    ) %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "same"
    ) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu",
      padding = "same"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  expect_error(LRP$new(converter,
    array(rnorm(4 * 32 * 31, 3), dim = c(4, 32, 31, 3)),
    channels_first = FALSE
  ))

  # Simple Rule
  lrp_simple <-
    LRP$new(converter, data, channels_first = FALSE, ignore_last_act = FALSE)
  expect_equal(dim(lrp_simple$get_result()), c(4, 32, 32, 3, 2))
  expect_true(
    lrp_simple$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Epsilon Rule
  lrp_eps_default <- LRP$new(converter, data,
    rule_name = "epsilon",
    dtype = "double",
    channels_first = FALSE
  )
  expect_equal(dim(lrp_eps_default$get_result()), c(4, 32, 32, 3, 2))
  expect_true(
    lrp_eps_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_eps_1 <- LRP$new(converter, data,
    rule_name = "epsilon",
    rule_param = 1,
    channels_first = FALSE,
    ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_eps_1$get_result()), c(4, 32, 32, 3, 2))
  expect_true(
    lrp_eps_1$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(converter, data,
    rule_name = "epsilon",
    dtype = "double",
    channels_first = FALSE,
    ignore_last_act = FALSE
  )
  expect_equal(dim(lrp_ab_default$get_result()), c(4, 32, 32, 3, 2))
  expect_true(
    lrp_ab_default$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  lrp_ab_2 <- LRP$new(converter, data,
    rule_name = "epsilon",
    rule_param = 2,
    channels_first = FALSE
  )
  expect_equal(dim(lrp_ab_2$get_result()), c(4, 32, 32, 3, 2))
  expect_true(
    lrp_ab_2$get_result(type = "torch.tensor")$dtype == torch_float()
  )
})


test_that("LRP: Keras model with two inputs + two outputs", {
  library(keras)

  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
  auxiliary_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 2, activation = 'softmax', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 4, activation = 'tanh') %>%
    layer_dense(units = 2, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(auxiliary_output, main_output)
  )

  converter <- Converter$new(model)
  data <- lapply(list(c(5), c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))

  lrp <- LRP$new(converter, data, channels_first = FALSE,
                       output_idx = list(c(2), c(1,3)))
  result <- lrp$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))

  lrp_eps <- LRP$new(converter, data, channels_first = FALSE, rule_name = "epsilon",
                         output_idx = list(c(1), c(1,2)))
  result <- lrp_eps$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))

  lrp_ab <- LRP$new(converter, data, channels_first = FALSE, rule_name = "alpha_beta",
                    rule_param = 0.5, output_idx = list(c(2), c(2, 3)))
  result <- lrp_ab$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))
})


test_that("LRP: Correctness (CNN)", {
  library(keras)
  library(torch)

  data <- torch_tensor(array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3)) * 5,
                       dtype = torch_double())

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus",
      padding = "valid", use_bias = FALSE
    ) %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "valid", use_bias = FALSE
    ) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu",
      padding = "valid", use_bias = FALSE
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 16, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE)

  # test non-fitted model
  converter <- Converter$new(model, dtype = "double")

  lrp <- LRP$new(converter, data, channels_first = FALSE, dtype = "double")
  res <- converter$model(data, channels_first = FALSE, save_last_layer = TRUE)
  out <- converter$model$modules_list[[7]]$preactivation
  lrp_result_sum <-
    lrp$get_result(type = "torch.tensor")$sum(dim = c(2, 3, 4))
  expect_lt(as.array(mean(abs(lrp_result_sum - out)^2)), 1e-10)

  lrp <-
    LRP$new(converter, data, channels_first = FALSE, ignore_last_act = FALSE,
            dtype = "double")
  res <- converter$model(data, channels_first = FALSE, save_last_layer = TRUE)
  out <- converter$model$modules_list[[7]]$output - 0.5
  lrp_result_no_last_act_sum <-
    lrp$get_result(type = "torch.tensor")$sum(dim = c(2, 3, 4))
  expect_lt(as.array(mean(abs(lrp_result_no_last_act_sum - out)^2)), 1e-10)
})


test_that("LRP: Correctness (mixed model with add layer)", {
  library(keras)
  library(torch)

  data <- lapply(list(c(12,15,3), c(20), c(10)),
                 function(x) torch_randn(c(10,x), dtype = torch_double()))

  input_1 <- layer_input(shape = c(12,15,3))
  part_1 <- input_1 %>%
    layer_conv_2d(3, c(4,4), activation = "relu", use_bias = FALSE) %>%
    layer_conv_2d(2, c(3,3), activation = "relu", use_bias = FALSE) %>%
    layer_flatten() %>%
    layer_dense(12, activation = "relu", use_bias = FALSE)
  input_2 <- layer_input(shape = c(10))
  part_2 <- input_2 %>%
    layer_dense(12, activation = "relu", use_bias = FALSE)
  input_3 <- layer_input(shape = c(20))
  part_3 <- input_3 %>%
    layer_dense(12, activation = "relu", use_bias = FALSE)

  output <- layer_add(c(part_1, part_3, part_2)) %>%
    layer_dense(10, activation = "relu", use_bias = FALSE) %>%
    layer_dense(1, activation = "linear", use_bias = FALSE)

  model <- keras_model(
    inputs = c(input_1, input_3, input_2),
    outputs = output
  )

  conv <- Converter$new(model)

  lrp <- LRP$new(conv, data, channels_first = FALSE, dtype = "double")

  res_total_true <- as.array(model(lapply(data, as.array)))
  res <- lrp$result[[1]]
  res_total <- as.array(
    res[[1]]$sum(c(2,3,4,5)) + res[[2]]$sum(c(2,3)) + res[[3]]$sum(c(2,3)))

  expect_lt(mean((res_total - res_total_true)^2), 1e-12)
})

test_that("LRP: Correctness (mixed model with concat layer)", {
  library(keras)
  library(torch)

  data <- lapply(list(c(12,15,3), c(20), c(10)),
                 function(x) torch_randn(c(10,x)))

  input_1 <- layer_input(shape = c(12,15,3))
  part_1 <- input_1 %>%
    layer_conv_2d(3, c(4,4), activation = "relu", use_bias = FALSE) %>%
    layer_conv_2d(2, c(3,3), activation = "relu", use_bias = FALSE) %>%
    layer_flatten() %>%
    layer_dense(20, activation = "relu", use_bias = FALSE)
  input_2 <- layer_input(shape = c(10))
  part_2 <- input_2 %>%
    layer_dense(50, activation = "tanh", use_bias = FALSE)
  input_3 <- layer_input(shape = c(20))
  part_3 <- input_3 %>%
    layer_dense(40, activation = "relu", use_bias = FALSE)

  output <- layer_concatenate(c(part_1, part_3, part_2)) %>%
    layer_dense(100, activation = "relu", use_bias = FALSE) %>%
    layer_dense(1, activation = "linear", use_bias = FALSE)

  model <- keras_model(
    inputs = c(input_1, input_3, input_2),
    outputs = output
  )

  conv <- Converter$new(model)

  lrp <- LRP$new(conv, data, channels_first = FALSE)

  res_total_true <- as.array(model(lapply(data, as.array)))
  res <- lrp$result[[1]]
  res_total <- as.array(
    res[[1]]$sum(c(2,3,4,5)) + res[[2]]$sum(c(2,3)) + res[[3]]$sum(c(2,3)))

  expect_lt(mean((res_total - res_total_true)^2), 1e-10)
})
