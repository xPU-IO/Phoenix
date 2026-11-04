// SPDX-License-Identifier: Apache-2.0
#include "phx_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

// phoenix C API header
#include <phoenix.h>

namespace {

constexpr size_t PAGE_SIZE = 4096;

inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

}  // namespace

PhxLoader::PhxLoader(int device_id)
    : device_id_(device_id), initialized_(false) {
    /* Public Phoenix device ids are CUDA/HIP/NPU ordinals. */
    dev_ = device_id;

    int ret = phxfs_open(dev_);
    if (ret < 0) {
        throw std::runtime_error("PhxLoader: phxfs_open(" +
                                 std::to_string(dev_) + ") failed with " +
                                 std::to_string(ret));
    }
    initialized_ = true;
}

PhxLoader::~PhxLoader() {
    if (initialized_) {
        try {
            close();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

uintptr_t PhxLoader::regmem(void *gpu_ptr, size_t size) {
    // Align size up to device page size, required by phxfs_regmem
    size_t page_size = phxfs_get_page_size();
    size_t aligned_size = align_up(size, page_size);

    void *target_addr = nullptr;
    int ret = phxfs_regmem(dev_, gpu_ptr, aligned_size, &target_addr);
    if (ret < 0) {
        throw std::runtime_error(
            "PhxLoader::regmem: phxfs_regmem failed with " +
            std::to_string(ret) + ", size=" + std::to_string(size) +
            ", aligned_size=" + std::to_string(aligned_size));
    }

    auto key = reinterpret_cast<uintptr_t>(gpu_ptr);
    reg_map_[key] = {aligned_size, reinterpret_cast<uintptr_t>(target_addr)};

    return reinterpret_cast<uintptr_t>(target_addr);
}

void PhxLoader::deregmem(void *gpu_ptr, size_t size) {
    auto key = reinterpret_cast<uintptr_t>(gpu_ptr);
    auto it = reg_map_.find(key);
    if (it == reg_map_.end()) {
        return;  // Not registered, nothing to do
    }

    size_t aligned_size = it->second.first;
    int ret = phxfs_deregmem(dev_, gpu_ptr, aligned_size);
    if (ret < 0) {
        // Log but don't throw in cleanup path
        fprintf(stderr,
                "PhxLoader::deregmem: phxfs_deregmem failed with %d\n", ret);
    }

    reg_map_.erase(it);
}

void PhxLoader::load_tensors_into_buffer(
    const std::string &path, uintptr_t gpu_ptr,
    const std::vector<std::tuple<off_t, off_t, size_t>> &batch) {

    // Verify gpu_ptr is registered
    auto it = reg_map_.find(gpu_ptr);
    if (it == reg_map_.end()) {
        throw std::runtime_error(
            "PhxLoader::load_tensors_into_buffer: gpu_ptr " +
            std::to_string(gpu_ptr) + " not registered");
    }

    if (batch.empty()) {
        return;  // Nothing to read
    }

    // Open file with O_DIRECT for DMA
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::system_error(errno, std::system_category(),
                                "PhxLoader::load_tensors_into_buffer: open(" +
                                    path + ")");
    }

    try {
        // Serial phxfs_read for each batch item.
        // Each item is (buf_offset, file_offset, nbytes), all 4K-aligned
        // by the Python side (build_read_groups).
        for (const auto &item : batch) {
            off_t buf_offset = std::get<0>(item);
            off_t file_offset = std::get<1>(item);
            size_t nbytes = std::get<2>(item);

            auto t_dma_start = std::chrono::steady_clock::now();

            ssize_t result = phxfs_read(
                fd, dev_,
                reinterpret_cast<void *>(gpu_ptr),
                buf_offset,
                static_cast<ssize_t>(nbytes),
                file_offset);

            auto t_dma_end = std::chrono::steady_clock::now();
            dma_time_ns_.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_dma_end - t_dma_start).count(),
                std::memory_order_relaxed);

            if (result < 0) {
                throw std::runtime_error(
                    "PhxLoader::load_tensors_into_buffer: phxfs_read failed "
                    "with " + std::to_string(result) + ", path=" + path +
                    ", buf_offset=" + std::to_string(buf_offset) +
                    ", file_offset=" + std::to_string(file_offset) +
                    ", nbytes=" + std::to_string(nbytes));
            }

            // Tolerate short reads for the tail padding region.
            // The actual tensor data ends before file_size, which is within
            // the readable range. Short reads only occur when aligned_end
            // exceeds file_size (4K alignment padding at file tail).
            if (static_cast<size_t>(result) < nbytes) {
                // Short read: only warn, don't throw. The Python side
                // ensures tensor data is within [file_offset, file_size].
                fprintf(stderr,
                        "PhxLoader::load_tensors_into_buffer: short read "
                        "(expected %zu, got %zd) at file_offset=%lld, "
                        "path=%s — likely file tail padding, continuing\n",
                        nbytes, result,
                        (long long)file_offset, path.c_str());
            }
        }

        ::close(fd);
    } catch (...) {
        ::close(fd);
        throw;
    }
}

void PhxLoader::load_tensors_into_buffer_async(
    const std::string &path, uintptr_t gpu_ptr,
    const std::vector<std::tuple<off_t, off_t, size_t>> &batch) {
    // If a previous async DMA is still running, wait for it first.
    if (dma_future_.valid()) {
        dma_future_.wait();
    }
    // Launch DMA in a C++ thread (not affected by Python GIL).
    // Capture copies of path/batch/gpu_ptr (path and batch are copied by value
    // to avoid lifetime issues).
    dma_future_ = std::async(std::launch::async, [this, path, gpu_ptr, batch]() {
        this->load_tensors_into_buffer(path, gpu_ptr, batch);
    });
}

void PhxLoader::wait_dma() {
    if (dma_future_.valid()) {
        dma_future_.wait();
        dma_future_.get();  // rethrow any exception from the async task
    }
}

double PhxLoader::get_dma_seconds() const {
    return static_cast<double>(dma_time_ns_.load(std::memory_order_relaxed))
           / 1e9;
}

void PhxLoader::reset_dma_timer() {
    dma_time_ns_.store(0, std::memory_order_relaxed);
}

void PhxLoader::close() {
    if (!initialized_) {
        return;
    }

    // Deregister any remaining registered memory
    for (const auto &[gpu_ptr, info] : reg_map_) {
        void *ptr = reinterpret_cast<void *>(gpu_ptr);
        int ret = phxfs_deregmem(dev_, ptr, info.first);
        if (ret < 0) {
            fprintf(stderr,
                    "PhxLoader::close: phxfs_deregmem failed with %d\n", ret);
        }
    }
    reg_map_.clear();

    int ret = phxfs_close(dev_);
    if (ret < 0) {
        fprintf(stderr, "PhxLoader::close: phxfs_close failed with %d\n",
                ret);
    }
    initialized_ = false;
}
