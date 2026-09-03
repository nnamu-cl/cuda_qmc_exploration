#ifndef HARNESS_DEVICE_MEMORY_H_
#define HARNESS_DEVICE_MEMORY_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

#include "harness/timing.h"

namespace qmc::harness {

struct DeviceDeleter {
  void operator()(void* pointer) const noexcept { cudaFree(pointer); }
};

template <class T>
using DeviceUnique = std::unique_ptr<T, DeviceDeleter>;

template <class T>
[[nodiscard]] DeviceUnique<T> DeviceAlloc(std::size_t count) {
  T* pointer = nullptr;
  CheckCudaCall(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
  return DeviceUnique<T>(pointer);
}

template <class T>
void DeviceFill(const DeviceUnique<T>& memory, int byte_value,
                std::size_t count) {
  CheckCudaCall(cudaMemset(memory.get(), byte_value, count * sizeof(T)),
                "cudaMemset");
}

}  // namespace qmc::harness

#endif
