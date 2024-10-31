#include <torch/torch.h>
#include <c10/xpu/XPUFunctions.h>

int main(int argc, const char* argv[]) 
{
  torch::DeviceType device;
  int num_devices = 0;
  if (torch::xpu::is_available()) {
    std::cout << "XPU devices detected" << std::endl;
    device = torch::kXPU;

    num_devices = torch::xpu::device_count();
    std::cout << "Number of XPU devices: " << num_devices << std::endl;

    
    for (int i = 0; i < num_devices; ++i) {
      c10::xpu::set_device(i);
      std::cout << "Device " << i << ":" << std::endl;
      //std::string device_name = c10::xpu::get_device_name();
      //std::cout << "Device " << i << ": " << device_name << std::endl;

      c10::xpu::DeviceProp device_prop{};
      c10::xpu::get_device_properties(&device_prop, i);
      std::cout << "  Name: " << device_prop.name << std::endl;
      std::cout << "  Total memory: " << device_prop.global_mem_size / (1024 * 1024) << " MB" << std::endl;
    }
  } else {
    device = torch::kCPU;
    std::cout << "No XPU devices detected, setting device to CPU" << std::endl;
  }
   
  return 0;
}
