//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <vector>
#include <string>

#include <dirent.h>

#include <opencv2/opencv.hpp>

#include "dll/rbm.hpp"
#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"
#include "dll/ocv_visualizer.hpp"
#include "cpp_utils/data.hpp"

namespace {

std::vector<cv::Mat> read_images(const std::string& dataset_path, const std::string& category){
    std::vector<cv::Mat> images;

    auto file_path = dataset_path + "/" + category;

    struct dirent* entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".jpg") != file_name.size() - 4) {
            continue;
        }

        std::string full_name(file_path + "/" + file_name);

        images.push_back(cv::imread(full_name, CV_LOAD_IMAGE_ANYDEPTH));

        if (!images.back().data) {
            std::cout << "Impossible to read image " << full_name << std::endl;
            return {};
        }
    }

    return images;
}

std::vector<cv::Mat> clean_images(const std::vector<cv::Mat>& images, std::size_t width, std::size_t height){
    std::vector<cv::Mat> cleaned_images;

    for(auto& image : images){
        std::size_t image_width = image.size().width;
        std::size_t image_height = image.size().height;

        if(image_width == width && image_height == height){
            cleaned_images.push_back(image);
        } else {
            long diff_x = -(long(image_width) - width);
            long diff_y = -(long(image_height) - height);

            cv::Mat proper_image(cv::Size(width, height), image.type());
            proper_image = cv::Scalar(255);

            cv::Rect source_roi(0, 0, image_width + (diff_x > 0 ? 0 : diff_x), image_height + (diff_y > 0 ? 0 : diff_y));
            cv::Rect target_roi(0, 0, image_width + (diff_x > 0 ? 0 : diff_x), image_height + (diff_y > 0 ? 0 : diff_y));
            image(source_roi).copyTo(proper_image(target_roi));

            cleaned_images.push_back(proper_image);
        }
    }

    return cleaned_images;
}

std::vector<etl::dyn_matrix<float, 3>> convert_images(const std::vector<cv::Mat>& images, std::size_t width, std::size_t height){
    std::vector<etl::dyn_matrix<float, 3>> converted_images;

    for(auto& image : images){
        converted_images.emplace_back(1UL, height, width);

        auto& conv_image = converted_images.back();

        for(std::size_t y = 0; y < height; ++y){
            for(std::size_t x = 0; x < width; ++x){
                conv_image(0UL, y, x) = static_cast<float>(image.at<uint8_t>(cv::Point(x, y)));
            }
        }

        //for(auto& pixel : conv_image){
            //pixel = pixel < 128 ? 0.0 : 1.0f;
        //}

        cpp::normalize(conv_image);
    }

    return converted_images;
}

} //end of anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: caltech dataset_path" << std::endl;
        return -1;
    }

    std::vector<std::string> args;

    for (std::size_t i = 1; i < static_cast<size_t>(argc); ++i) {
        args.emplace_back(argv[i]);
    }

    std::string dataset_path = args[0];
    std::string category = "car_side";

    constexpr const std::size_t width = 294;
    constexpr const std::size_t height = 198;

    auto images = read_images(dataset_path, category);
    std::cout << images.size() << " images read" << std::endl;

    auto proper_images = clean_images(images, width, height);
    std::cout << proper_images.size() << " images cleaned" << std::endl;

    auto training_images = convert_images(proper_images, width, height);
    std::cout << training_images.size() << " images converted" << std::endl;

    training_images.resize(100);

    //using layer_1_t = dll::conv_rbm_desc<1, width, height, 24, (width - 10 + 1), (height - 10 + 1),
    using layer_1_t = dll::conv_rbm_mp_desc<1, width, height, 24, (width - 10 + 1), (height - 10 + 1), 3,
          dll::batch_size<50>,
          dll::momentum,
          //dll::parallel_mode,
          dll::weight_decay<dll::decay_type::L2>,
          dll::sparsity<dll::sparsity_method::LEE>,
          dll::bias<dll::bias_mode::SIMPLE>,
          //dll::verbose,
          dll::visible<dll::unit_type::GAUSSIAN>,
          dll::watcher<dll::opencv_rbm_visualizer>
              >::layer_t;

    std::cout << "Size of layer 1:" << sizeof(layer_1_t) / 1024 / 1024 << "MB" << std::endl;

    auto layer_1 = std::make_unique<layer_1_t>();

    layer_1->initial_momentum = 0.5;
    layer_1->final_momentum = 0.9;
    layer_1->learning_rate = 1e-5;
    layer_1->pbias = 0.002;
    layer_1->pbias_lambda = 250;
    layer_1->l2_weight_cost = 0.01;

    //layer_1->learning_rate *= 3;
    //layer_1->sparsity_target = 0.05;
    //layer_1->sparsity_cost *= 0.1;
    //layer_1->

    //layer_1->learning_rate *= 10; //Batch size

    layer_1->display();
    layer_1->train(training_images, 500);
    layer_1->store("layer_1.dat");

    return 0;
}
