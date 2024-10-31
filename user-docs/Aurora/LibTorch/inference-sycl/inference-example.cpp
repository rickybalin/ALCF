#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include "sycl/sycl.hpp"
#include <vector>

const int N_BATCH = 1;
const int N_CHANNELS = 3;
const int N_PIXELS = 224;
const int INPUTS_SIZE = N_BATCH*N_CHANNELS*N_PIXELS*N_PIXELS;
const int OUTPUTS_SIZE = N_BATCH*N_CHANNELS;

int main(int argc, const char* argv[]) {
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(argv[1]);
    std::cout << "Loaded the model\n";
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
    
  model.to(torch::Device(torch::kXPU));
  std::cout << "Model offloaded to GPU\n\n";

  // Create the input data on the host
  std::vector<float> inputs(INPUTS_SIZE);
  srand(12345);
  for (int i=0; i<INPUTS_SIZE; i++) {
    inputs[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }
  std::cout << "Generated input data on the host \n\n";

  // Move input data to the device with SYCL
  sycl::queue Q(sycl::gpu_selector_v);
  std::cout << "SYCL running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n\n";
  float *d_inputs = sycl::malloc_device<float>(INPUTS_SIZE, Q);
  Q.memcpy((void *) d_inputs, (void *) inputs.data(), INPUTS_SIZE*sizeof(float));
  Q.wait();

  // Pre-allocate the output array on device and fill with a number
  double *d_outputs = sycl::malloc_device<double>(OUTPUTS_SIZE, Q);
  Q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(OUTPUTS_SIZE, [=](sycl::id<1> idx) {
      d_outputs[idx] = 1.2345;
    });
  });
  Q.wait();
  std::cout << "Offloaded input data to the GPU \n\n";

  // Convert input array to Torch tensor
  auto options = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(torch::kXPU);
  torch::Tensor input_tensor = torch::from_blob(
                                 d_inputs,
                                 {N_BATCH,N_CHANNELS,N_PIXELS,N_PIXELS},
                                 options);
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == torch::kXPU);
  std::cout << "Created the input Torch tesor on GPU\n\n";

  // Perform inference
  torch::NoGradGuard no_grad; // equivalent to "with torch.no_grad():" in PyTorch
  torch::Tensor output = model.forward({input_tensor}).toTensor();
  std::cout << "Performed inference\n\n";

  // Copy the output Torch tensor to the SYCL pointer
  auto output_tensor_ptr = output.contiguous().data_ptr();
  Q.memcpy((void *) d_outputs, (void *) output_tensor_ptr, OUTPUTS_SIZE*sizeof(double));
  Q.wait();
  std::cout << "Copied output Torch tensor to SYCL pointer\n";

  return 0;
}
