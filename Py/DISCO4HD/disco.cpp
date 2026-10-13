#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
namespace py = pybind11;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using CArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Same centered-sum-of-products update as R src/parallel_disco.cpp:
// SS_new = SS + n/(n+1) * (x-mu)'(x-mu). Pearson normalization uses SS directly.
py::array_t<double> disco_optimized_cpp(CArray d, CArray ref, CArray weights) {
    const auto a=d.request(), r=ref.request(), w=weights.request();
    if (a.ndim!=2 || r.ndim!=2 || w.ndim!=2)
        throw std::invalid_argument("Inputs must be 2D arrays");
    const auto n=r.shape[0], p=r.shape[1];
    if (n<3 || p<2 || a.shape[1]!=p || w.shape[0]!=p || w.shape[1]!=p)
        throw std::invalid_argument("Invalid reference or weight dimensions");
    // forcecast converts F-order and strided callers; row-major maps respect
    // NumPy's actual field layout and keep converted inputs alive throughout.
    const Eigen::Map<const RowMatrix> data(static_cast<double*>(a.ptr),a.shape[0],p);
    const Eigen::Map<const RowMatrix> reference(static_cast<double*>(r.ptr),n,p);
    const Eigen::Map<const RowMatrix> weight(static_cast<double*>(w.ptr),p,p);
    if (!data.allFinite() || !reference.allFinite() || !weight.allFinite() || (weight.array()<0).any())
        throw std::invalid_argument("Inputs must be finite and weights nonnegative");
    const Eigen::RowVectorXd mu=reference.colwise().mean();
    const Eigen::MatrixXd centered=reference.rowwise()-mu;
    const Eigen::MatrixXd ss=centered.transpose()*centered;
    const Eigen::VectorXd sd=ss.diagonal().array().sqrt();
    if (!ss.allFinite() || (sd.array()<=0).any())
        throw std::invalid_argument("Reference biomarkers must have positive finite variance");
    const Eigen::MatrixXd original=ss.array()/(sd*sd.transpose()).array();
    // NumPy owns this allocation; no temporary Eigen pointer escapes.
    py::array_t<double> result({static_cast<py::ssize_t>(a.shape[0]),static_cast<py::ssize_t>(1)});
    auto output=result.mutable_unchecked<2>();
    for (py::ssize_t i=0;i<a.shape[0];++i) {
        const Eigen::RowVectorXd delta=data.row(i)-mu;
        const Eigen::MatrixXd updated=ss+(static_cast<double>(n)/(n+1.0))*delta.transpose()*delta;
        const Eigen::VectorXd sn=updated.diagonal().array().sqrt();
        if (!updated.allFinite() || (sn.array()<=0).any())
            throw std::runtime_error("Numerical failure in updated correlation matrix");
        const Eigen::MatrixXd correlation=updated.array()/(sn*sn.transpose()).array();
        const Eigen::MatrixXd difference=original-correlation;
        const double value=(difference.array().square()*weight.array()).sum();
        if (!std::isfinite(value) || value<0)
            throw std::runtime_error("Numerical failure in weighted perturbation");
        output(i,0)=std::log(value*static_cast<double>(n)*n); // zero -> -Inf, as in R
    }
    return result;
}
PYBIND11_MODULE(disco_optimized,m) {
    m.def("disco_optimized_cpp",&disco_optimized_cpp,"Compute DISCO with the R-compatible rank-one Pearson update");
}
