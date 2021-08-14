#include "v5.h"
#include "opencv2/opencv.hpp"

int main(int argc, char **argv)
{
    std::cout<<"1"<<std::endl;
    YoloV5Detector YoloInstance;
    YoloInstance.InitModel("../ng0801.engine");

    auto t_start_pre = std::chrono::high_resolution_clock::now();
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    
    for(int i = 0; i < 100; i++){
        cv::Mat big_img = cv::imread("../test.jpg", -1);

        std::vector<RecBox> res;

        t_start_pre = std::chrono::high_resolution_clock::now();

        YoloInstance.Process(big_img, res);
        
        t_end_pre = std::chrono::high_resolution_clock::now();
        total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
        std::cout << "end to end take: " << total_pre << " ms." << std::endl;

        for (size_t n_i = 0; n_i < res.size(); n_i++)
        {
            std::cout << "predict class is: " << res[n_i].class_id << std::endl;
            cv::rectangle(big_img, res[n_i].rect, cv::Scalar(255), 10, 1, 0);   
        }
        cv::imwrite("cv_image.jpg", big_img);

    }

    
    return 0;
}
