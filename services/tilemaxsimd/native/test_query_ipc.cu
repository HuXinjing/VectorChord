// This software is licensed under AGPLv3 or Elastic License v2.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

struct VctmGpu;
struct VctmIpcQuery;
extern "C" int vctm_gpu_create(int, size_t, size_t, VctmGpu **, char *, size_t);
extern "C" void vctm_gpu_destroy(VctmGpu *);
extern "C" int vctm_gpu_upload_batch(VctmGpu *, const uint64_t *,
                                      const unsigned char *const *, const size_t *,
                                      size_t, char *, size_t);
extern "C" int vctm_gpu_score(VctmGpu *, const unsigned char *, size_t, uint32_t,
                               uint32_t, uint8_t, uint8_t, const uint64_t *,
                               const uint32_t *, size_t, float *, char *, size_t);
extern "C" int vctm_gpu_open_ipc_fp16(
    VctmGpu *, const cudaIpcMemHandle_t *, const cudaIpcEventHandle_t *, size_t,
    uint32_t, uint32_t, VctmIpcQuery **, char *, size_t);
extern "C" int vctm_gpu_score_ipc_fp16(
    VctmGpu *, const VctmIpcQuery *, const uint64_t *, const uint32_t *, size_t,
    float *, char *, size_t);
extern "C" int vctm_gpu_close_ipc_fp16(VctmIpcQuery *, char *, size_t);

namespace {
constexpr uint32_t dims = 320;
constexpr uint32_t rows = 21;
constexpr uint32_t document_rows = 8;
constexpr size_t mib = 1024 * 1024;
using Query = std::array<half, rows * dims>;
using Document = std::array<half, document_rows * dims>;
struct Handles {
  cudaIpcMemHandle_t allocation;
  cudaIpcEventHandle_t ready;
};

void check(cudaError_t value) {
  if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}

void check(int value, const char *error) {
  if (value != 0) throw std::runtime_error(error);
}

template <typename Array>
Array fixture(double phase) {
  Array result{};
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = __float2half(static_cast<float>(std::sin(i * 0.123 + phase) / 20));
  return result;
}

void check_pipe(ssize_t size) {
  if (size != sizeof(Handles)) throw std::runtime_error("IPC test pipe failed");
}

void consume(int fd, int device) {
  Handles handles{};
  check_pipe(read(fd, &handles, sizeof(handles)));
  close(fd);
  const auto query = fixture<Query>(0.5);
  const auto document = fixture<Document>(0.1);
  const auto other = fixture<Document>(4.0);
  char error[512]{};
  VctmGpu *raw_gpu = nullptr;
  check(vctm_gpu_create(device, 8 * mib, 4 * mib, &raw_gpu, error, sizeof(error)), error);
  struct Gpu {
    VctmGpu *ptr;
    ~Gpu() { vctm_gpu_destroy(ptr); }
  } gpu{raw_gpu};
  const uint64_t offsets[] = {0, sizeof(document)};
  const uint32_t counts[] = {document_rows, document_rows};
  const auto *query_bytes = reinterpret_cast<const unsigned char *>(query.data());
  const auto payloads = std::array{
      reinterpret_cast<const unsigned char *>(document.data()),
      reinterpret_cast<const unsigned char *>(other.data()),
  };
  const size_t lengths[] = {sizeof(document), sizeof(other)};
  check(vctm_gpu_upload_batch(gpu.ptr, offsets, payloads.data(), lengths, 2, error,
                              sizeof(error)), error);
  float expected[2]{}, actual[2]{};
  check(vctm_gpu_score(gpu.ptr, query_bytes, sizeof(query), rows, dims, 2, 1,
                       offsets, counts, 2, expected, error, sizeof(error)), error);
  VctmIpcQuery *invalid = nullptr;
  if (vctm_gpu_open_ipc_fp16(gpu.ptr, &handles.allocation, &handles.ready,
                              sizeof(query) - 1, rows, dims, &invalid, error,
                              sizeof(error)) == 0 || invalid != nullptr)
    throw std::runtime_error("incorrect query byte count accepted");
  struct Mapping {
    VctmIpcQuery *ptr = nullptr;
    ~Mapping() { vctm_gpu_close_ipc_fp16(ptr, nullptr, 0); }
  } mapping;
  const auto open_start = std::chrono::steady_clock::now();
  check(vctm_gpu_open_ipc_fp16(gpu.ptr, &handles.allocation, &handles.ready,
                              sizeof(query), rows, dims, &mapping.ptr, error,
                              sizeof(error)), error);
  const auto open_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - open_start).count();
  constexpr auto repetitions = 10;
  double ipc_total = 0;
  double host_total = 0;
  for (int i = 0; i < repetitions; ++i) {
    auto start = std::chrono::steady_clock::now();
    check(vctm_gpu_score_ipc_fp16(gpu.ptr, mapping.ptr, offsets, counts, 2,
                                  actual, error, sizeof(error)), error);
    ipc_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (std::memcmp(expected, actual, sizeof(expected)) != 0)
      throw std::runtime_error("IPC scores differ from host-input scores");
    start = std::chrono::steady_clock::now();
    check(vctm_gpu_score(gpu.ptr, query_bytes, sizeof(query), rows, dims, 2, 1,
                         offsets, counts, 2, expected, error, sizeof(error)), error);
    host_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  std::printf("consumer_gpu=%d queries=%d scores_equal=true ipc_open_ms=%.3f ipc_mean_ms=%.3f host_mean_ms=%.3f\n",
              device, repetitions, open_ms, ipc_total / repetitions, host_total / repetitions);
}

int produce(const char *executable, int source, int target) {
  int fds[2];
  if (pipe(fds) != 0) throw std::runtime_error("pipe creation failed");
  const auto child = fork();
  if (child < 0) throw std::runtime_error("fork failed");
  if (child == 0) {
    close(fds[1]);
    const auto read_fd = std::to_string(fds[0]);
    const auto device = std::to_string(target);
    execl(executable, executable, "consume", read_fd.c_str(), device.c_str(), nullptr);
    _exit(127);
  }
  close(fds[0]);
  int result = 0;
  try {
    check(cudaSetDevice(source));
    if (source != target) {
      int can_access = 0;
      check(cudaDeviceCanAccessPeer(&can_access, target, source));
      if (!can_access) throw std::runtime_error("CUDA peer access unavailable");
    }
    struct Exported {
      void *allocation = nullptr;
      cudaEvent_t ready = nullptr;
      ~Exported() {
        if (ready) cudaEventDestroy(ready);
        if (allocation) cudaFree(allocation);
      }
    } buffer;
    check(cudaMalloc(&buffer.allocation, 2 * mib));
    check(cudaEventCreateWithFlags(&buffer.ready, cudaEventInterprocess | cudaEventDisableTiming));
    const auto query = fixture<Query>(0.5);
    check(cudaMemcpy(buffer.allocation, query.data(), sizeof(query), cudaMemcpyHostToDevice));
    check(cudaEventRecord(buffer.ready));
    Handles handles{};
    check(cudaIpcGetMemHandle(&handles.allocation, buffer.allocation));
    check(cudaIpcGetEventHandle(&handles.ready, buffer.ready));
    check_pipe(write(fds[1], &handles, sizeof(handles)));
    close(fds[1]);
    fds[1] = -1;
    if (waitpid(child, &result, 0) != child) throw std::runtime_error("waitpid failed");
  } catch (...) {
    if (fds[1] != -1) close(fds[1]);
    waitpid(child, nullptr, 0);
    throw;
  }
  return WIFEXITED(result) ? WEXITSTATUS(result) : 1;
}
} // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc == 4 && std::strcmp(argv[1], "consume") == 0) {
      consume(std::stoi(argv[2]), std::stoi(argv[3]));
      return 0;
    }
    if (argc == 3) return produce(argv[0], std::stoi(argv[1]), std::stoi(argv[2]));
    std::fprintf(stderr, "usage: %s PRODUCER_GPU CONSUMER_GPU\n", argv[0]);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
