#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> generate_precence_map(const py::array_t<uint8_t> &_img) {
    auto img = _img.unchecked<2>();  // Accessing the numpy array as a 2D array
    ptrdiff_t rows = img.shape(0);
    ptrdiff_t cols = img.shape(1);
    
    // Initialize the precence map with zeros
    py::array_t<double> precence_map = py::array_t<double>({rows, cols});
    auto prec_map = precence_map.mutable_unchecked<2>();  // Accessing the numpy array as a 2D array (mutable)
    
    // Calculate the presence map
    for (ptrdiff_t i = 0; i < rows; i += 2) {
        for (ptrdiff_t j = 0; j < cols; j += 2) {
            double value = img(i, j) / 255.0;
            for (ptrdiff_t di = -10; di <= 10; ++di) {
                for (ptrdiff_t dj = -10; dj <= 10; ++dj) {
                    ptrdiff_t ni = i + di;
                    ptrdiff_t nj = j + dj;
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        prec_map(ni, nj) += value;
                    }
                }
            }
        }
    }

    return precence_map;
}

PYBIND11_MODULE(precence_map_module, m) {
    m.def("generate_precence_map", &generate_precence_map, "A function that generates and returns a presence map");
}
