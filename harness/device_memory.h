#ifndef HARNESS_DEVICE_MEMORY_H_
#define HARNESS_DEVICE_MEMORY_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

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

inline void CopyHostToDevice(void* dst, const void* src, std::size_t bytes) {
  CheckCudaCall(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
}

inline void CopyDeviceToHost(void* dst, const void* src, std::size_t bytes) {
  CheckCudaCall(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H");
}

template <class T>
void CopyHostToDevice(const DeviceUnique<T>& dst, std::span<const T> src) {
  CopyHostToDevice(dst.get(), src.data(), src.size_bytes());
}

template <class T>
void CopyDeviceToHost(std::span<T> dst, const DeviceUnique<T>& src) {
  CopyDeviceToHost(dst.data(), src.get(), dst.size_bytes());
}

template <class T>
[[nodiscard]] DeviceUnique<T> DeviceFromHost(std::span<const T> src) {
  DeviceUnique<T> memory = DeviceAlloc<T>(src.size());
  CopyHostToDevice(memory, src);
  return memory;
}

template <class T>
[[nodiscard]] std::vector<T> HostFromDevice(const DeviceUnique<T>& src,
                                            std::size_t count) {
  std::vector<T> out(count);
  CopyDeviceToHost(std::span<T>(out), src);
  return out;
}

}  // namespace qmc::harness

#endif
