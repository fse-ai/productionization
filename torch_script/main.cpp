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
    image = imread("/mnt/c/Users/Vishnu/Pictures/Screenshots/Screenshot (1).png", IMREAD_COLOR);
    string module_string = "traced_transforms.pt";
    
    if (image.empty()) {
        cout << "Could not read the image" << endl;
        return -1;
    } else {
        torch::jit::script::Module module;
        try {
            module = torch::jit::load(module_string);
        } catch (const c10:: Error& e) {
            cerr << "Error loading the module" << endl;
            return -1;
        }
        int width = 250;
        int height = 250;

        resize(image, image, Size(width, height), 0, 0, 1);

        torch::Tensor image_tensor = torch::from_blob(image.data, {width, height, image.channels()});
        image_tensor = image_tensor.permute({2, 0, 1});
        image_tensor = image_tensor.unsqueeze_(0);

        vector<torch::jit::IValue> input_tensor;
        input_tensor.push_back(image_tensor);

        torch::Tensor output = module.forward(input_tensor).toTensor();
        cout << output.size(1) << endl;
    }
    return 0;
}
