#' @include InterpretingLayer.R
#'
NULL

#' One-dimensional Convolution Layer of a Neural Network
#'
#' Implementation of a one-dimensional Convolutional Neural Network layer as a
#' \code{\link[torch]{nn_conv1d}} module where input, preactivation and output
#' values of the last forward pass are stored (same for a reference input,
#' if this is needed). Applies the torch function
#' \code{\link[torch]{nnf_conv1d}} for forwarding an input through a 1d
#' convolution followed by an activation function \eqn{\sigma} to the input
#' data, i.e.
#' \deqn{
#' y= \sigma(\code{nnf_conv1d(x,W,b)})
#' }
#'
#' @param weight The weight matrix of dimension \emph{(out_channels,
#' in_channels, kernel_length)}
#' @param bias The bias vector of dimension \emph{(out_channels)}
#' @param dim_in The input dimension of the layer: \emph{(in_channels,
#' in_length)}
#' @param dim_out The output dimensions of the layer: \emph{(out_channels,
#' out_length)}
#' @param stride The stride used in the convolution, by default `1`
#' @param padding The padding of the layer, by default `c(0,0)` (left, right),
#' can be an integer or a two-dimensional tuple
#' @param dilation The dilation of the layer, by default `1`
#' @param activation_name The name of the activation function used, by
#' default `"linear"`
#' @param dtype The data type of all the parameters (Use `'float'` or
#' `'double'`)
#'
#' @section Attributes:
#' \describe{
#'   \item{`self$W`}{The weight matrix of this layer with shape
#'     \emph{(out_channels, in_channels, kernel_length)}}
#'   \item{`self$b`}{The bias vector of this layer with shape
#'     \emph{(out_channels)}}
#'   \item{`self$...`}{Many attributes are inherited from the superclass
#'     [InterpretingLayer], e.g. `input`, `input_dim`, `preactivation`,
#'     `activation_name`, etc.}
#' }
#'
#' @noRd
#'
conv1d_layer <- nn_module(
  classname = "Conv1D_Layer",
  inherit = InterpretingLayer,

  #
  # weight: [out_channels, in_channels, kernel_length]
  # bias  : [out_channels]
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = c(0, 0),
                        dilation = 1,
                        activation_name = "linear",
                        dtype = "float") {

    # [in_channels, in_length]
    self$input_dim <- dim_in
    # [out_channels, out_length]
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_length <- dim(weight)[-c(1, 2)]
    self$stride <- stride
    self$padding <- padding # [left, right]
    self$dilation <- dilation

    self$get_activation(activation_name)

    if (!inherits(weight, "torch_tensor")) {
      self$W <- torch_tensor(weight)
    } else {
      self$W <- weight
    }
    if (!inherits(bias, "torch_tensor")) {
      self$b <- torch_tensor(bias)
    } else {
      self$b <- bias
    }
    self$set_dtype(dtype)
  },


  #' @section `self$forward()`:
  #' The forward function takes an input and forwards it through the layer,
  #' updating the the values of `input`, `preactivation` and `output`
  #'
  #' ## Usage
  #' `self(x)`
  #'
  #' ## Arguments
  #' \describe{
  #' \item{`x`}{The input torch tensor of dimensions \emph{(batch_size,
  #' in_channels, in_length)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the layer with respect to the given inputs,
  #' with dimensions \emph{(batch_size, out_channels, out_length)}
  #'
  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE) {
    if (save_input) {
      self$input <- x
    }
    # Pad the input
    x <- nnf_pad(x, pad = self$padding)
    # Apply conv1d
    preactivation <- nnf_conv1d(x, self$W,
      bias = self$b,
      stride = self$stride,
      padding = 0,
      dilation = self$dilation
    )
    if (save_preactivation) {
      self$preactivation <- preactivation
    }

    output <- self$activation_f(preactivation)
    if (save_output) {
      self$output <- output
    }

    output
  },

  #' @section `self$update_ref()`:
  #' This function takes the reference input and runs it through
  #' the layer, updating the the values of `input_ref`, `preactivation_ref`
  #' and `output_ref`
  #'
  #' ## Usage
  #' `self$update_ref(x_ref)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x_ref`}{The new reference input, of dimensions \emph{(1,
  #'   in_channels, in_length)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the reference input after
  #' passing through the layer, of dimension \emph{(1, out_channels,
  #' out_length)}
  #'
  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    # Apply padding
    x_ref <- nnf_pad(x_ref, pad = self$padding)
    # Apply conv1d
    preactivation_ref <- nnf_conv1d(x_ref, self$W,
      bias = self$b,
      stride = self$stride,
      padding = 0,
      dilation = self$dilation
    )
    if (save_preactivation) {
      self$preactivation_ref <- preactivation_ref
    }

    output_ref <- self$activation_f(preactivation_ref)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },



  #' @section `self$get_input_relevances()`:
  #' This method uses the output layer relevances and calculates the input
  #' layer relevances using the specified rule.
  #'
  #' ## Usage
  #' `self$get_input_relevances(`\cr
  #' `  rel_output,`\cr
  #' `  rule_name = 'simple',` \cr
  #' `  rule_param = NULL)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`rel_output`}{The output relevances, of dimensions
  #'     \emph{(batch_size, out_channels, out_length, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the relevance
  #'     scores are calculated. Implemented are `"simple"`, `"epsilon"`,
  #'     `"alpha_beta"`, `"ww"` (default: `"simple"`).}
  #'   \item{`rule_param`}{The parameter of the selected rule. Note: Only the
  #'   rules `"epsilon"` and `"alpha_beta"` take use of the parameter. Use the
  #'   default value `NULL` for the default parameters (`"epsilon"` :
  #'   \eqn{0.01}, `"alpha_beta"` : \eqn{0.5}).}
  #' }
  #'
  #' ## Return
  #' Returns the relevance score of the layer's input to the model output as a
  #' torch tensor of size \emph{(batch_size, in_channels, in_length,
  #' model_out)}
  #'
  #'
  get_input_relevances = function(rel_output,
                                  rule_name = "simple",
                                  rule_param = NULL) {
    if (rule_name == "simple") {
      z <- self$preactivation$unsqueeze(4)
      # add a small stabilizer
      z <- z + (z == 0) * 1e-12

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(4)
    } else if (rule_name == "epsilon") {
      # set default parameter
      if (is.null(rule_param)) {
        epsilon <- 0.001
      } else {
        epsilon <- rule_param
      }

      z <- self$preactivation$unsqueeze(4)
      z <- z + epsilon * torch_sgn(z) + (z == 0) * 1e-12

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(4)
    } else if (rule_name == "alpha_beta") {
      # set default parameter
      if (is.null(rule_param)) {
        alpha <- 0.5
      } else {
        alpha <- rule_param
      }

      # Get positive and negative part of the output
      out_part <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)
      input <- self$input$unsqueeze(4)

      # Apply simple rule on the positive part
      z <- rel_output /
        (out_part$pos + (out_part$pos == 0) * 1e-16)$unsqueeze(4)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_pos <- t1 * (input * (input > 0)) + t2 * (input * (input <= 0))

      # Apply simple rule on the negative part
      z <- rel_output /
        (out_part$neg + (out_part$neg == 0) * 1e-16)$unsqueeze(4)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # Calculate over all relevance for the lower layer
      rel_input <- rel_pos * alpha + rel_neg * (1 - alpha)
    }

    rel_input
  },

  #' @section `self$get_input_multiplier()`:
  #' This function is the local implementation of the DeepLift method for this
  #' layer and returns the multiplier from the input contribution to the
  #' output.
  #'
  #' ## Usage
  #' `self$get_input_multiplier(mult_output, rule_name = "rescale")`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`mult_output`}{The multiplier of the layer output contribution
  #'   to the model output. A torch tensor of shape
  #'   \emph{(batch_size, out_channels, out_length, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the multiplier is
  #'        calculated. Implemented are `"rescale"` and `"reveal_cancel"`
  #'        (default: `"rescale"`).}
  #' }
  #'
  #' ## Return
  #' Returns the contribution multiplier of the layer's input to the model
  #' output as torch tensor of dimension \emph{(batch_size, in_channels,
  #' in_length, model_out)}.
  #'
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {

        # output       [batch_size, out_channels, out_length]
        # delta_output [batch_size, out_channels, out_length, 1]
        delta_output <- (self$output - self$output_ref)$unsqueeze(4)
        delta_preact <-
          (self$preactivation - self$preactivation_ref)$unsqueeze(4)

        nonlin_mult <- delta_output /
          (delta_preact + 1e-16 * (delta_preact == 0))

        # mult_output   [batch_size, out_channels, out_length, model_out]
        # nonlin_mult   [batch_size, out_channels, out_length, 1]
        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult
      } else if (rule_name == "reveal_cancel") {
        pos_and_neg_output <-
          self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_x_pos <- pos_and_neg_output$pos
        delta_x_neg <- pos_and_neg_output$neg

        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref

        delta_output_pos <-
          0.5 * (act(x_ref + delta_x_pos) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x_neg))

        delta_output_neg <-
          0.5 * (act(x_ref + delta_x_neg) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x_pos))

        mult_pos <- mult_output *
          (delta_output_pos /
            (delta_x_pos + (delta_x_pos == 0) * 1e-16))$unsqueeze(4)
        mult_neg <- mult_output *
          (delta_output_neg /
            (delta_x_neg - (delta_x_neg == 0) * 1e-16))$unsqueeze(4)
      }
    }

    #
    # -------------- Linear part -----------------------
    #

    # input        [batch_size, in_channels, in_length]
    # delta_input  [batch_size, in_channels, in_length, 1]
    delta_input <- (self$input - self$input_ref)$unsqueeze(4)

    # weight      [out_channels, in_channels, kernel_length]
    weight <- self$W

    # mult_input    [batch_size, in_channels, in_length, model_out]
    mult_input <-
      self$get_gradient(mult_pos, weight * (weight > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, weight * (weight < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), weight) *
        (delta_input == 0)

    mult_input
  },

  #' @section `self$get_gradient()`:
  #' This method uses \code{\link[torch]{nnf_conv_transpose1d}} to multiply
  #' the input with the gradient of a layer's output with respect to the
  #' layer's input. This results in the gradients of the model output with
  #' respect to layer's input.
  #'
  #' ## Usage
  #' `self$get_gradient(input, weight)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`input`}{The gradients of the upper layer, a tensor of dimension
  #'   \emph{(batch_size, out_channels, out_length, model_out)}}
  #'   \item{`weight`}{A weight tensor of dimensions \emph{(out_channels,
  #'   in_channels, kernel_length)}}
  #' }
  #'
  #' ## Return
  #' Returns the gradient of the model's output with respect to the layer input
  #' as a torch tensor of dimension \emph{(batch_size, in_channels, in_length,
  #' model_out)}.
  #'
  get_gradient = function(input, weight) {

    # Since we have added the model_out dimension, strides and dilation need to
    # be extended by 1.

    stride <- c(self$stride, 1)

    # dilation is a number or a tuple of length 2
    dilation <- c(self$dilation, 1)

    out <- nnf_conv_transpose2d(input, weight$unsqueeze(4),
      bias = NULL,
      stride = stride,
      padding = 0,
      dilation = dilation
    )

    # If stride is > 1, it could happen that the reconstructed input after
    # padding (out) lost some dimensions, because multiple input shapes are
    # mapped to the same output shape. Therefore, we use padding with zeros to
    # fill in the missing irrelevant input values.
    lost_length <-
      self$input_dim[2] + self$padding[1] + self$padding[2] - dim(out)[3]

    out <- nnf_pad(out, pad = c(0, 0, 0, lost_length))
    # Now we have added the missing values such that
    # dim(out) = dim(padded_input)

    # Apply the inverse padding to obtain dim(out) = dim(input)
    out <- out[, , (self$padding[1] + 1):(dim(out)[3] - self$padding[2]), ]

    out
  },

  #' @section `self$get_pos_and_neg_outputs()`:
  #' This method separates the linear layer output (i.e. the preactivation)
  #' into the positive and negative parts.
  #'
  #' ## Usage
  #' `self$get_pos_and_neg_outputs(input, use_bias = FALSE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`input`}{The input whose linear output we want to decompose into
  #'   the positive and negative parts}
  #'   \item{`use_bias`}{Boolean whether the bias vector should be considered
  #'   (default: FALSE)}
  #' }
  #'
  #' ## Return
  #' Returns a decomposition of the linear output of this layer with input
  #' `input` into the positive and negative parts. A list of two torch
  #' tensors with size \emph{(batch_size, out_channels, out_length)} and
  #' keys `$pos` and `$neg`
  #'
  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias == TRUE) {
      b_pos <- self$b * (self$b > 0) * 0.5
      b_neg <- self$b * (self$b <= 0) * 0.5
    } else {
      b_pos <- NULL
      b_neg <- NULL
    }

    conv1d <- function(x, W, b) {
      x <- nnf_pad(x, pad = self$padding)
      out <- nnf_conv1d(x, W,
        bias = b,
        stride = self$stride,
        padding = 0,
        dilation = self$dilation
      )
      out
    }

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <-
      conv1d(input * (input > 0), self$W * (self$W > 0), b_pos) +
      conv1d(input * (input < 0), self$W * (self$W < 0), b_pos)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <-
      conv1d(input * (input > 0), self$W * (self$W < 0), b_neg) +
      conv1d(input * (input < 0), self$W * (self$W > 0), b_neg)

    output
  },


  #' @section `self$set_dtype()`:
  #' This function changes the data type of the weight and bias tensor to be
  #' either `"float"` or `"double"`.
  #'
  #' ## Usage
  #' `self$set_dtype(dtype)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`dtype`}{The data type of the layer's parameters. Use `"float"` or
  #'   `"double"`}
  #' }
  #'
  set_dtype = function(dtype) {
    if (dtype == "float") {
      self$W <- self$W$to(torch_float())
      self$b <- self$b$to(torch_float())
    } else if (dtype == "double") {
      self$W <- self$W$to(torch_double())
      self$b <- self$b$to(torch_double())
    } else {
      stop("Unknown argument for 'dtype' : '", dtype, "'. ",
           "Use 'float' or 'double' instead!")
    }
    self$dtype <- dtype
  }
)