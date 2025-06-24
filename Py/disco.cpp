#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;

Eigen::MatrixXd compute_disco(
    const Eigen::MatrixXd& d,
    const Eigen::MatrixXd& d_ref,
    const Eigen::MatrixXd& weight) {
    
    const int n0 = d_ref.rows();
    const int p = d_ref.cols();  // 虽然未使用但保留声明
    Eigen::MatrixXd results(d.rows(), 1);    
    
    // 计算参考数据的均值和协方差
    Eigen::RowVectorXd mu = d_ref.colwise().mean();
    Eigen::MatrixXd centered = d_ref.rowwise() - mu;
    Eigen::MatrixXd cov_mat = (centered.adjoint() * centered) / (n0 - 1.0);
    Eigen::VectorXd s = cov_mat.diagonal().array().sqrt();
    
    // 并行计算每个样本的DISCO值
    #pragma omp parallel for
    for (int k = 0; k < d.rows(); ++k) {
        try {
            Eigen::RowVectorXd x = d.row(k);
            Eigen::RowVectorXd delta = x - mu;
            Eigen::RowVectorXd delta_mu = delta / (n0 + 1.0);
            
            // 更新协方差矩阵
            Eigen::MatrixXd delta_outer = delta.adjoint() * delta;
            Eigen::MatrixXd delta_mu_outer = delta_mu.adjoint() * delta_mu;
            Eigen::MatrixXd sum_new = (n0 - 1.0) * cov_mat + delta_outer - n0 * delta_mu_outer;
            Eigen::MatrixXd cov_prime = sum_new / n0;
            
            // 计算新标准差
            Eigen::VectorXd var_prime = cov_prime.diagonal();
            var_prime = var_prime.array().max(1e-10);
            Eigen::VectorXd s_prime = var_prime.array().sqrt();
            
            // 计算相关系数矩阵
            Eigen::MatrixXd s_outer = s_prime * s_prime.adjoint();
            Eigen::MatrixXd cc1 = cov_prime.array() / s_outer.array();
            cc1 = (cc1.array().isNaN()).select(0, cc1);
            
            Eigen::MatrixXd original_corr = cov_mat.array() / (s * s.adjoint()).array();
            Eigen::MatrixXd diff = original_corr - cc1;
            
            // 计算加权差异
            double sum_val = (diff.array().square() * weight.array()).sum();
            results(k, 0) = (sum_val > 0) ? std::log(sum_val * n0 * n0) : std::log(1e-300);
        } catch (...) {
            results(k, 0) = NAN;
        }
    }
    
    return results;
}

// Pybind11 绑定接口
py::array_t<double> disco_optimized_cpp(
    py::array_t<double>& d,
    py::array_t<double>& d_ref,
    py::array_t<double>& weight) {
    
    // 将NumPy数组转换为Eigen矩阵
    auto buf_d = d.request();
    auto buf_ref = d_ref.request();
    auto buf_weight = weight.request();
    
    if (buf_d.ndim != 2 || buf_ref.ndim != 2 || buf_weight.ndim != 2)
        throw std::runtime_error("Input must be 2D arrays");
    
    Eigen::Map<Eigen::MatrixXd> mat_d(
        static_cast<double*>(buf_d.ptr), 
        buf_d.shape[0], buf_d.shape[1]);
    
    Eigen::Map<Eigen::MatrixXd> mat_ref(
        static_cast<double*>(buf_ref.ptr), 
        buf_ref.shape[0], buf_ref.shape[1]);
    
    Eigen::Map<Eigen::MatrixXd> mat_weight(
        static_cast<double*>(buf_weight.ptr), 
        buf_weight.shape[0], buf_weight.shape[1]);
    
    // 调用计算函数
    Eigen::MatrixXd results = compute_disco(mat_d, mat_ref, mat_weight);
    
    // 返回NumPy数组
     std::vector<size_t> shape = {(size_t)results.rows(), 1};
    std::vector<size_t> strides = {sizeof(double), sizeof(double) * results.rows()};
    
    py::buffer_info buf(
        results.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,  // 维度
        shape,
        strides
    );
    
    return py::array_t<double>(buf);
}

// 模块定义
PYBIND11_MODULE(disco_optimized, m) {
    m.def("disco_optimized_cpp", &disco_optimized_cpp, 
          "Calculate DISCO values (optimized C++ implementation)");
}
