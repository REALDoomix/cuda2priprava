// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

void cu_swap_image(CudaImg& img1, CudaImg& img2, int2 pos);
void cu_insert_mask(CudaImg& big, CudaImg& small, int2 pos, uchar3 mask, bool is_and_mask);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    /*if (argc < 4) {
        printf("Použití: %s obrazek1.jpg obrazek2.jpg obrazek3.jpg\n", argv[0]);
        return 1;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    cv::Mat img3 = cv::imread(argv[3], cv::IMREAD_COLOR);

    if (!img1.data || !img2.data || !img3.data) {
        printf("Chyba při načítání obrázků!\n");
        return 1;
    }

    CudaImg img1_cuda(img1);
    CudaImg img2_cuda(img2);
    CudaImg img3_cuda(img3);

    // === ČÁST 1: SWAP dvou pruhů (3× každý) ===
    int strip_height = img1.rows / 5;
    cv::Mat buffer(strip_height, img1.cols, CV_8UC3);
    CudaImg buffer_cuda(buffer);

    // První pruh (horní)
    for (int i = 0; i < 3; ++i) {
        int2 pos = {0, 0};
        cu_swap_image(img1_cuda, buffer_cuda, pos);
        cu_swap_image(buffer_cuda, img2_cuda, pos);
        cu_swap_image(img1_cuda, buffer_cuda, pos);
    }


    // === ČÁST 2: INSERT MASK ve tvaru čtyřlístku ===

    // Vyříznutí části z image3
    int small_width = 400;
    int small_height = 400;
    int start_x = std::min(200, img3.cols - small_width - 200);
    int start_y = std::min(200, img3.rows - small_height - 200);

    cv::Rect roi(start_x, start_y, small_width, small_height);
    cv::Mat small_img = img3(roi).clone();
    CudaImg small_cuda(small_img);

    // Pozice vložení
    int2 pos1 = {0, 0};
    int2 pos2 = {img3.cols - small_width, 0};
    int2 pos3 = {0, img3.rows - small_height};
    int2 pos4 = {img3.cols - small_width, img3.rows - small_height};

    // Vložení 4× s různými barevnými maskami
    cu_insert_mask(img3_cuda, small_cuda, pos1, {1, 1, 1}, false);      // originál
    cu_insert_mask(img3_cuda, small_cuda, pos2, {1, 0, 0}, false);      // jen červená
    cu_insert_mask(img3_cuda, small_cuda, pos3, {0, 1, 0}, false);      // jen zelená
    cu_insert_mask(img3_cuda, small_cuda, pos4, {0, 0, 1}, false);      // jen modrá
*/

if (argc < 3) {
    printf("Usage: %s background.jpg dandelion.png\n", argv[0]);
    return 1;
}

cv::Mat background = cv::imread(argv[1], cv::IMREAD_COLOR);
cv::Mat dandelion = cv::imread(argv[2], cv::IMREAD_UNCHANGED); // Load with alpha channel

if (!background.data || !dandelion.data) {
    printf("Error loading images!\n");
    return 1;
}

CudaImg background_cuda(background);
CudaImg dandelion_cuda(dandelion);

srand(time(0)); // Seed for random number generation

while (true) {
    // Generate random height for the dandelion
    int random_height = rand() % 201 + 100; // Random height in range [100, 300]
    int random_width = (random_height * dandelion.cols) / dandelion.rows; // Maintain aspect ratio

    // Resize dandelion
    cv::Mat resized_dandelion(random_height, random_width, CV_8UC4);
    CudaImg resized_dandelion_cuda(resized_dandelion);
    cu_scale(dandelion_cuda, resized_dandelion_cuda);

    // Generate random position
    int x_pos = rand() % (background.cols - random_width);
    int y_pos = rand() % (background.rows - random_height);

    // Insert resized dandelion into the background
    cu_insert_image(background_cuda, resized_dandelion_cuda, {x_pos, y_pos}, {1, 1, 1}, false);

    // Overlay text above the dandelion
    std::string text = "Position: (" + std::to_string(x_pos) + ", " + std::to_string(y_pos) + ")";
    cv::putText(background, text, cv::Point(x_pos, y_pos - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    // Display the result
    cv::imshow("Dandelions in Nature", background);

    // Wait for key press
    char key = cv::waitKey(0);
    if (key == 'q') break;
}

    // Rotate img1 by 90 degrees clockwise
    cv::Mat rotated_img(img1.cols, img1.rows, CV_8UC3); // Swapped dimensions
    CudaImg rotated_cuda(rotated_img);

    cu_rotate90(img1_cuda, rotated_cuda, 1); // Rotate clockwise

    // Copy the rotated image back to host memory for display
    rotated_img = cv::Mat(rotated_img.rows, rotated_img.cols, CV_8UC3, rotated_cuda.m_p_uchar3);

    // Display the rotated image
    cv::imshow("Obrazek 1 po rotaci", rotated_img);


    // Resize img1 to half its size
    cv::Mat resized_img(img1.rows / 2, img1.cols / 2, CV_8UC3);
    CudaImg resized_cuda(resized_img);

    cu_scale(img1_cuda, resized_cuda);

    // Copy the resized image back to host memory for display
    resized_img = cv::Mat(resized_img.rows, resized_img.cols, CV_8UC3, resized_cuda.m_p_uchar3);

    // Display the resized image
    cv::imshow("Obrazek 1 po zmene velikosti", resized_img);



    // Inicializace fontu
    initialize_font();

    // Vykreslení znaku 'A' na pozici (50, 50) s červenou barvou
    uchar3 red = {255, 0, 0};
    cu_draw_char(img1_cuda, 50, 50, 'A', red);

    // Zobrazení výsledného obrázku
    cv::imshow("Obrazek s vykreslenym znakem", img1);

    // === Zobrazení výsledků ===
    cv::imshow("Obrazek 1 po swapu", img1);
    cv::imshow("Obrazek 2 po swapu", img2);
    cv::imshow("Buffer", buffer);
    cv::imshow("Obrazek 3 s maskovanym vkladem", img3);

    cv::waitKey(0);
    return 0;
}

