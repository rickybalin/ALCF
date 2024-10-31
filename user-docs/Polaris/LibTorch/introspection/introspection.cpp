#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>

int main(int argc, const char* argv[]) 
{
  torch::DeviceType device;
  int num_devices = 0;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA devices detected" << std::endl;
    device = torch::kCUDA;

    num_devices = torch::cuda::device_count();
    std::cout << "Number of CUDA devices: " << num_devices << std::endl;

    /*
    for (int i = 0; i < num_devices; ++i) {
      c10::cuda::set_device(i);
      std::cout << "Device " << i << ":" << std::endl;

      c10::cuda::DeviceProp device_prop{};
      c10::cuda::get_device_properties(&device_prop, i);
      std::cout << "  Name: " << device_prop.name << std::endl;
      std::cout << "  Total memory: " << device_prop.global_mem_size / (1024 * 1024) << " MB" << std::endl;
    }*/
  } else {
    device = torch::kCPU;
    std::cout << "No CUDA devices detected, setting device to CPU" << std::endl;
  }
   
  return 0;
}
