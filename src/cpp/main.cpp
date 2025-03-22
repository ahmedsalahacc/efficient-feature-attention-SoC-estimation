#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <chrono>

using namespace std;

void test_model(torch::jit::script::Module model)
{
    cout << "Testing the model\n";
    torch::Device device(torch::kCPU);
    model.to(device);
    model.eval();

    torch::Tensor input = torch::randn({1, 5});
    input = input.to(device);

    // record the time taken to run the model
    auto start = chrono::steady_clock::now();
    auto output = model.forward({input}).toTensor();
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Time taken to run the model: " << duration.count() << "ms\n";
}

int main(int argc, const char *argv[])
{
    torch::jit::script::Module model;
    try
    {
        model = torch::jit::load("deployed_model.pt");
        test_model(model);
    }
    catch (const c10::Error &e)
    {
        cerr << "Error loading the model\n";
        return -1;
    }

    cout << "Model loaded successfully\n";

    return 0;
}