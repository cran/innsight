% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/InterpretingMethod.R
\name{get_result}
\alias{get_result}
\title{Get the result of an interpretation method}
\usage{
get_result(x, ...)
}
\arguments{
\item{x}{An object of the class \code{\link{InterpretingMethod}} including the
subclasses \code{\link{Gradient}}, \code{\link{SmoothGrad}}, \code{\link{LRP}}, \code{\link{DeepLift}},
\code{\link{DeepSHAP}}, \code{\link{IntegratedGradient}}, \code{\link{ExpectedGradient}} and
\code{\link{ConnectionWeights}}.}

\item{...}{Other arguments specified in the R6 method
\code{InterpretingMethod$get_result()}. See \code{\link{InterpretingMethod}} for details.}
}
\description{
This is a generic S3 method for the R6 method
\code{InterpretingMethod$get_result()}. See the respective method described in
\code{\link{InterpretingMethod}} for details.
}
