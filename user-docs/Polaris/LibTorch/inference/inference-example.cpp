#include <torch/torch.h>
#include <torch/script.h>

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
    
  model.to(torch::Device(torch::kCUDA));
  std::cout << "Model offloaded to GPU\n\n";

  auto options = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(torch::kCUDA);
  torch::Tensor input_tensor = torch::rand({1,3,224,224}, options);
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == torch::kCUDA);
  std::cout << "Created the input tesor on GPU\n";

  torch::Tensor output = model.forward({input_tensor}).toTensor();
  std::cout << "Performed inference\n\n";

  std::cout << "Slice of predicted tensor is : \n";
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

  return 0;
}
