// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phx_loader.h"

namespace py = pybind11;

PYBIND11_MODULE(_phxloader, m) {
    m.doc() = "Phoenix DMA loader for direct storage";

    py::class_<PhxLoader>(m, "PhxLoader")
        .def(py::init<int>(),
             py::arg("device_id"))
        .def("regmem",
             [](PhxLoader &self, uintptr_t gpu_ptr, size_t size) -> uintptr_t {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 return self.regmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Register GPU memory for DMA. Returns CPU mapped address.")
        .def("deregmem",
             [](PhxLoader &self, uintptr_t gpu_ptr, size_t size) {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 self.deregmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Deregister GPU memory previously registered with regmem().")
        .def("load_tensors_into_buffer",
             [](PhxLoader &self, const std::string &path,
                uintptr_t gpu_ptr,
                const std::vector<std::tuple<off_t, off_t, size_t>> &batch) {
                 self.load_tensors_into_buffer(path, gpu_ptr, batch);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("path"), py::arg("gpu_ptr"), py::arg("batch"),
             "Batch read multiple (buf_offset, file_offset, nbytes) entries "
             "from a single file into the registered GPU buffer.")
        .def("load_tensors_into_buffer_async",
             [](PhxLoader &self, const std::string &path,
                uintptr_t gpu_ptr,
                const std::vector<std::tuple<off_t, off_t, size_t>> &batch) {
                 self.load_tensors_into_buffer_async(path, gpu_ptr, batch);
             },
             py::arg("path"), py::arg("gpu_ptr"), py::arg("batch"),
             "Async batch read: launches DMA in a C++ thread, returns "
             "immediately. Use wait_dma() to join.")
        .def("wait_dma",
             [](PhxLoader &self) { self.wait_dma(); },
             py::call_guard<py::gil_scoped_release>(),
             "Wait for the most recent load_tensors_into_buffer_async to "
             "complete.")
        .def("get_dma_seconds",
             [](PhxLoader &self) { return self.get_dma_seconds(); },
             "Get accumulated pure DMA time in seconds (C++ steady_clock).")
        .def("reset_dma_timer",
             [](PhxLoader &self) { self.reset_dma_timer(); },
             "Reset the DMA timer to zero.")
        .def("close", &PhxLoader::close, "Close the phxfs device.");
}
