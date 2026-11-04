// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Forward declarations for phoenix C API
extern "C" {
int phxfs_open(int device_id);
int phxfs_close(int device_id);
int phxfs_find_dev(int device_id);
uint64_t phxfs_get_page_size(void);
ssize_t phxfs_read(int fd, int device_id, void *buf, off_t buf_offset,
                    ssize_t nbyte, off_t f_offset);
int phxfs_regmem(int device_id, const void *addr, size_t len,
                 void **target_addr);
int phxfs_deregmem(int device, const void *addr, size_t len);
}

class PhxLoader {
public:
    explicit PhxLoader(int device_id);
    ~PhxLoader();

    // Non-copyable
    PhxLoader(const PhxLoader &) = delete;
    PhxLoader &operator=(const PhxLoader &) = delete;

    /// Register GPU memory for DMA. Returns CPU mapped address.
    /// Size is internally aligned up to 64K (HUGE_PAGE_SIZE).
    uintptr_t regmem(void *gpu_ptr, size_t size);

    /// Deregister GPU memory previously registered with regmem().
    void deregmem(void *gpu_ptr, size_t size);

    /// Batch read multiple (buf_offset, file_offset, nbytes) entries from
    /// a single file into the registered GPU buffer. Each entry's
    /// buf_offset, file_offset, and nbytes must be 4K-aligned (handled by
    /// the Python side via build_read_groups). Serial phxfs_read per entry.
    void load_tensors_into_buffer(
        const std::string &path, uintptr_t gpu_ptr,
        const std::vector<std::tuple<off_t, off_t, size_t>> &batch);

    /// Async version of load_tensors_into_buffer: launches DMA in a C++ thread
    /// (std::async) and returns immediately. Use wait_dma() to join.
    void load_tensors_into_buffer_async(
        const std::string &path, uintptr_t gpu_ptr,
        const std::vector<std::tuple<off_t, off_t, size_t>> &batch);

    /// Wait for the most recent load_tensors_into_buffer_async to complete.
    void wait_dma();

    /// Get accumulated pure DMA time (seconds) measured inside phxfs_read
    /// loops via steady_clock. Independent of Python main thread timing.
    double get_dma_seconds() const;

    /// Reset the DMA timer to zero.
    void reset_dma_timer();

    /// Close the phxfs device. Called automatically by destructor.
    void close();

private:
    int dev_;            // phxfs device id
    int device_id_;      // original vendor device id
    bool initialized_;

    // Tracks registered memory: gpu_ptr -> (aligned_size, cpu_target_addr)
    std::unordered_map<uintptr_t, std::pair<size_t, uintptr_t>> reg_map_;

    // Async DMA support
    std::future<void> dma_future_;

    // Pure DMA timing (accumulated across all async calls, thread-safe)
    std::atomic<long long> dma_time_ns_{0};
};
