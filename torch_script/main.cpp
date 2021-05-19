#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

using namespace cv;
using namespace std;

int main() {
    Mat image;
    image = imread("/mnt/c/Users/Vishnu/Pictures/airplane.jpg", IMREAD_COLOR);
    string model_path = "trace_model.pt";

    if (image.empty()) {
        cout << "Could not read the image" << endl;
        return -1;
    } else {
        torch::jit::script::Module model;
        try {
            model = torch::jit::load(model_path);
        } catch (const c10:: Error& e) {
            cerr << "Error loading the module" << endl;
            return -1;
        }
        int width = 32;
        int height = 32;

        resize(image, image, Size(width, height), 0, 0, 1);

        torch::Tensor image_tensor = torch::from_blob(image.data, {width, height, image.channels()});
        image_tensor = image_tensor.permute({2, 0, 1});
        image_tensor = image_tensor.unsqueeze_(0);

        torch::Tensor mean = image_tensor.mean({2, 3});
        cout << image_tensor << endl;

        vector<torch::jit::IValue> input_tensor;
        input_tensor.push_back(image_tensor);

        torch::Tensor model_output = model.forward(input_tensor).toTensor();
        cout << model_output.argmax() << endl;
    }
    return 0;
}
