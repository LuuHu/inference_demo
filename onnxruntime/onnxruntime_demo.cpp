#include <iostream>
#include "stdio.h"
#include "onnxruntime_cxx_api.h"

#include "opencv2/opencv.hpp"

#include "sys/time.h"


inline void print_TensorInfo(Ort::Unowned<Ort::TensorTypeAndShapeInfo>* tensor_info, char* head_s)
{
    printf("[%16s][%4d] : %s=[", __FUNCTION__, __LINE__, head_s);
    for (size_t i = 0; i < tensor_info->GetDimensionsCount(); i++)
    {
        printf("%ld", tensor_info->GetShape()[i]);
        if (i!=tensor_info->GetDimensionsCount()-1)
            printf(", ");
    }
    printf("] \n");
}


inline float time_interval(struct timeval old_time, struct timeval new_time)
{
    int sec = new_time.tv_sec - old_time.tv_sec;
    int usec = new_time.tv_usec - old_time.tv_usec;
    return sec + usec/1000000.;
}


int main(int, char **)
{
    std::string model_file = "../model/yolov7.onnx";
    std::string image_file = "../1.jpeg";
    // std::string image_file = "../1g.jpg";

    cv::Mat img = cv::imread(image_file);
    cv::resize(img, img, cv::Size(640,640));

    float* blob = new float[img.total()*3];
    for (int c = 0; c < 3; c++) 
        for (int  h = 0; h < img.rows; h++) 
            for (int w = 0; w < img.cols; w++) 
                blob[c * img.cols * img.rows + h * img.cols + w] = img.at<cv::Vec3b>(h, w)[c] / 255.0;


    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(4);
    Ort::Session session(env, model_file.c_str(), session_options); 


    auto ort_mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::AllocatorWithDefaultOptions ort_alloc;

    // print name/shape of inputs
    size_t input_count = session.GetInputCount();
    printf("[%16s][%4d] : input_count=(%ld)\n", __FUNCTION__, __LINE__, input_count);
    Ort::AllocatedStringPtr input_names = session.GetInputNameAllocated(input_count-1, ort_alloc);
    printf("[%16s][%4d] : input_names=(%s)\n", __FUNCTION__, __LINE__, input_names.get());
    // input_names.get_deleter();


    size_t output_count = session.GetInputCount();
    printf("[%16s][%4d] : output_count=(%ld)\n", __FUNCTION__, __LINE__, output_count);
    Ort::AllocatedStringPtr output_names = session.GetOutputNameAllocated(output_count-1, ort_alloc);
    printf("[%16s][%4d] : output_names=(%s)\n", __FUNCTION__, __LINE__, output_names.get());
    // output_names.get_deleter();

    auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto output_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    print_TensorInfo(&input_info, "input_shape");
    print_TensorInfo(&output_info, "output_shape");
    
    Ort::Value input_tensor = Ort::Value::CreateTensor(ort_mem, blob, img.total()*3, input_info.GetShape().data(), input_info.GetShape().size());
    assert(input_tensor.IsTensor());

    const char* input_node_names = input_names.get();
    const char* output_node_names = output_names.get();

    struct timeval tet, tet_r;
    gettimeofday(&tet, NULL);

    while (1)
    {
        
        cv::Mat img2 = img.clone();

        auto output_tensors = session.Run(Ort::RunOptions(nullptr), &input_node_names, &input_tensor, 1, &output_node_names, 1);

        gettimeofday(&tet_r, NULL);
        printf("[%16s][%4d] : Run_time=(%f)\n", __FUNCTION__, __LINE__, time_interval(tet, tet_r));
        tet = tet_r;
        
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());


        printf("[%16s][%4d] : [", __FUNCTION__, __LINE__ );
        for (size_t i = 0; i < output_tensors.front().GetTensorTypeAndShapeInfo().GetDimensionsCount(); i++)
        {
            printf("%ld", output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[i]);
            // if (i!= output_tensors.front().GetTensorTypeAndShapeInfo().GetDimensionsCount()-1)
            printf(", ");
        }
        printf("] \n");


        float* floatarr = output_tensors.front().GetTensorMutableData<float>();

        for (size_t tt=0; tt<output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[0]; tt++)
        {
            for (int i = 0; i < 7; i++)
                printf("S[%d]=  %f  ", i%7, floatarr[i+tt*7]);
            printf("\n");
            cv::rectangle(img2, cv::Rect(cv::Point(int(floatarr[tt*7+1]),int(floatarr[tt*7+2])), cv::Point(int(floatarr[tt*7+3]),int(floatarr[tt*7+4]))), cv::Scalar(0.255,255), 2);
        }
        
        cv::imshow("img2", img2);
        cv::waitKey(100);
    }

    
