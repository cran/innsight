## ---- include = FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(
  fig.dpi = ifelse(Sys.getenv("RENDER_PLOTLY", unset = 0) == 1, 400, 50),
  collapse = TRUE,
  comment = "#>",
  fig.align = "center",
  out.width = "90%",
  eval = requireNamespace("palmerpenguins") & requireNamespace("luz") &
    torch::torch_is_installed()
)

## ---- echo = FALSE----------------------------------------------------------------------
Sys.setenv(LANG = "en_US.UTF-8")
set.seed(111)
torch::torch_manual_seed(111)

## ---- warning=FALSE, fig.width=12, fig.height=8, message=FALSE--------------------------
library(palmerpenguins)

# remove NAs
penguin_data <- na.omit(penguins)
# create plot
GGally::ggpairs(penguin_data,
  ggplot2::aes(color = species, alpha = 0.75),
  columns = 3:7,
  progress = FALSE
)

## ----setup, fig.width=16----------------------------------------------------------------
# Load packages
library(torch)
library(luz)


# Create torch penguin dataset
penguin_dataset <- dataset(
  name = "penguin_dataset",
  initialize = function(df) {
    df <- na.omit(df) # remove NAs

    # Get all numeric features and transform them to `torch_tensor`
    x_cont <- df[, c(
      "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
      "body_mass_g", "year"
    )]
    x_cont <- torch_tensor(as.matrix(x_cont))

    # Get and encode (one-hot) categorical features and transform them to `torch_tensor`
    x_cat <- sapply(df[, c("island", "sex")], as.integer)
    x_cat <- torch_tensor(x_cat, dtype = torch_int64())
    x_cat <- torch_hstack(list(
      nnf_one_hot(x_cat[, 1]),
      nnf_one_hot(x_cat[, 2])
    ))

    # Stack and store all features together in the field 'x'
    self$x <- torch_hstack(list(x_cont, x_cat))

    # Get, transform and store the target variable (the three penguin species)
    self$y <- torch_tensor(as.integer(df$species))
  },
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i])
  },
  .length = function() {
    self$y$size()[[1]]
  }
)

## ---------------------------------------------------------------------------------------
# Normalize inputs
scaled_penguin_ds <- penguin_data
scaled_penguin_ds[, c(3:6, 8)] <- scale(penguin_data[, c(3:6, 8)])

# Create train and validation split
idx <- sample(seq_len(nrow(penguin_data)))
train_idx <- idx[1:250]
valid_idx <- idx[251:333]

# Create train and validation datasets
train_ds <- penguin_dataset(scaled_penguin_ds[train_idx, ])
valid_ds <- penguin_dataset(scaled_penguin_ds[valid_idx, ])

# Create dataloaders
train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE)

# We use the whole dataset as the test data
test_ds <- penguin_dataset(scaled_penguin_ds)

## ---------------------------------------------------------------------------------------
# Create the model
net <- nn_module(
  initialize = function(dim_in) {
    # Here, we define our sequential model
    self$seq_model <- nn_sequential(
      nn_linear(dim_in, 256),
      nn_relu(),
      nn_dropout(p = 0.4),
      nn_linear(256, 256),
      nn_relu(),
      nn_dropout(p = 0.4),
      nn_linear(256, 64),
      nn_relu(),
      nn_dropout(p = 0.4),
      nn_linear(64, 3),
      nn_softmax(dim = 2)
    )
  },

  # The forward pass should only contain the call of the sequential model
  forward = function(x) {
    self$seq_model(x)
  }
)

## ---------------------------------------------------------------------------------------
# We have imbalanced classes, so we weight the output classes accordingly
weight <- length(train_ds$y) /
  (3 * torch_stack(lapply(1:3, function(i) sum(train_ds$y == i))))

# Fit the model
fitted <- net %>%
  setup(
    loss =
      function(input, target) nnf_nll_loss(log(input), target, weight = weight),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>%
  set_hparams(dim_in = 10) %>%
  fit(train_dl, epochs = 50, valid_data = valid_dl)

# Extract the sequential model
model <- fitted$model$seq_model

# Show result
get_metrics(fitted)[c(99, 100, 199, 200), ]

## ---------------------------------------------------------------------------------------
library(innsight)

# Define input and output names
input_names <-
  c(
    "bill_length", "bill_depth", "flipper_length", "body_mass", "year",
    "island_Biscoe", "island_Dream", "island_Torgersen",
    "sex_female", "sex_male"
  )
output_names <- c("Adelie", "Chinstrap", "Gentoo")

# Create the `Converter` object
converter_1 <- Converter$new(model,
  input_dim = 10,
  input_names = input_names,
  output_names = output_names
)

## ---------------------------------------------------------------------------------------
# Create a second `Converter` object for combined categorical features
converter_2 <- Converter$new(model,
  input_dim = 10,
  output_names = output_names
)

## ---------------------------------------------------------------------------------------
# Data to be analyzed (in this case, we use the whole dataset)
data <- test_ds$x
# Apply method 'LRP' with rule alpha-beta
lrp_ab_1 <- LRP$new(converter_1, data, rule_name = "alpha_beta", rule_param = 2)

# the result for 333 instances, 10 inputs and all 3 outputs
dim(get_result(lrp_ab_1))

## ---- results='hide', message=FALSE-----------------------------------------------------
# Apply method as in the other case
lrp_ab_2 <- LRP$new(converter_2, data, rule_name = "alpha_beta", rule_param = 2)

# Adjust input dimension and input names in the method converter object
lrp_ab_2$converter$input_dim[[1]] <- 7
lrp_ab_2$converter$input_names[[1]][[1]] <-
  c(
    "bill_length", "bill_depth", "flipper_length", "body_mass", "year",
    "island", "sex"
  )

# Combine (sum) the results for feature 'island' and 'sex'
lrp_ab_2$result[[1]][[1]] <- torch_cat(list(
  lrp_ab_2$result[[1]][[1]][, 1:5, ], # results for all numeric features
  lrp_ab_2$result[[1]][[1]][, 6:8, ]$sum(2, keepdim = TRUE), # results for feature island
  lrp_ab_2$result[[1]][[1]][, 9:10, ]$sum(2, keepdim = TRUE) # results for feature sex
),
dim = 2
)

# Now we have the desired output shape with combined categorical features
dim(get_result(lrp_ab_2))
#> [1] 333  7  3

## ---- fig.width=8-----------------------------------------------------------------------
library(ggplot2)
# Separated categorical features
plot(lrp_ab_1, output_idx = c(1, 3)) +
  coord_flip()
# Combined categorical features
plot(lrp_ab_2, output_idx = c(1, 3)) +
  coord_flip()

## ---- echo = FALSE----------------------------------------------------------------------
knitr::kable(penguin_data[1, ])

## ---- fig.width=8-----------------------------------------------------------------------
library(ggplot2)

boxplot(lrp_ab_2,
  output_idx = c(1, 3), preprocess_FUN = identity,
  ref_data_idx = 1
) +
  coord_flip() +
  facet_grid(rows = vars(output_node))

