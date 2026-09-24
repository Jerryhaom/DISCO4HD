#' Report compiled backend capabilities
#' @return A list with a logical openmp field. With FALSE the C++ backend is serial.
#' @export
#' @examples
#' disco_backend_info()
disco_backend_info <- function() list(openmp = disco_openmp_enabled())
