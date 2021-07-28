#include "v5.h"

// static Logger gLogger;


YoloV5Detector::YoloV5Detector(){}

YoloV5Detector::~YoloV5Detector()
{
    // Destroy the engine
    m_context_->destroy();
    m_engine_->destroy();
    m_runtime_->destroy();
}


bool YoloV5Detector::InitModel(std::string engine_name, int gpu_id)
{
    cudaSetDevice(gpu_id);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    // initLibNvInferPlugins(&gLogger, "");

    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // IRuntime *runtime = createInferRuntime(gLogger);
    // m_runtime_ = createInferRuntime(gLogger);
    assert(m_runtime_ != nullptr);
    // ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    m_engine_ = m_runtime_->deserializeCudaEngine(trtModelStream, size);

    assert(m_engine_ != nullptr);
    // IExecutionContext *context = m_engine_->createExecutionContext();
    m_context_ = m_engine_->createExecutionContext();
    assert(m_context_ != nullptr);

    delete[] trtModelStream;

}


// void YoloV5Detector::Forward(cv::Mat& img, YoloV5DetectionResult *det_result)
int YoloV5Detector::Process(cv::Mat &img, std::vector<RecBox> &result)
{
    if (img.empty())
        return -1;

    cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row)
    {
        uchar *uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col)
        {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    auto start = std::chrono::system_clock::now();

    // Run inference
    doInference();

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::cout << "detect count " << counts[0] << std::endl;
    for (int j = 0; j < counts[0]; j++)
    {
        float *curBbox = boxes + j * 4;
        float *curScore = scores + j;
        float *curClass = classes + j;

        RecBox rec_box;
        rec_box.rect = get_rect(img, curBbox);
        rec_box.conf = *curScore;
        rec_box.class_id = int(*curClass);
        result.push_back(rec_box); 
    }

    return 0;
}


void YoloV5Detector::doInference()
{
    // const ICudaEngine &engine = context.getEngine();
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    assert(m_engine_->getNbBindings() == 5);
    void *buffers[5];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = m_engine_->getBindingIndex(INPUT_NAME);
    const int countIndex = m_engine_->getBindingIndex(OUTPUT_COUNTS);
    const int bboxIndex = m_engine_->getBindingIndex(OUTPUT_BOXES);
    const int scoreIndex = m_engine_->getBindingIndex(OUTPUT_SCORES);
    const int classIndex = m_engine_->getBindingIndex(OUTPUT_CLASSES);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[countIndex], BATCH_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&buffers[bboxIndex], BATCH_SIZE * KEEP_TOPK * 4 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[scoreIndex], BATCH_SIZE * KEEP_TOPK * sizeof(float)));
    CHECK(cudaMalloc(&buffers[classIndex], BATCH_SIZE * KEEP_TOPK * sizeof(float)));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], data, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    m_context_->enqueue(BATCH_SIZE, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(counts, buffers[countIndex], BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(boxes, buffers[bboxIndex], BATCH_SIZE * KEEP_TOPK * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(scores, buffers[scoreIndex], BATCH_SIZE * KEEP_TOPK * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(classes, buffers[classIndex], BATCH_SIZE * KEEP_TOPK * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[countIndex]));
    CHECK(cudaFree(buffers[bboxIndex]));
    CHECK(cudaFree(buffers[scoreIndex]));
    CHECK(cudaFree(buffers[classIndex]));
}