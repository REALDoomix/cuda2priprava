#pragma once

#include <opencv2/core/mat.hpp>
#include <cuda_runtime.h>

// Structure definition for exchanging data between Host and Device
struct CudaImg
{
    uint3 m_size;             // size of picture
    union {
        void   *m_p_void;     // data of picture
        uchar1 *m_p_uchar1;   // grayscale
        uchar3 *m_p_uchar3;   // BGR
        uchar4 *m_p_uchar4;   // BGRA
    };

    __device__ __host__ uchar1& at1(int y, int x) {
        return m_p_uchar1[y * m_size.x + x];
    }

    __device__ __host__ uchar3& at3(int y, int x) {
        return m_p_uchar3[y * m_size.x + x];
    }

    __device__ __host__ uchar4& at4(int y, int x) {
        return m_p_uchar4[y * m_size.x + x];
    }

    __host__ CudaImg(cv::Mat& mat) {
        m_size.x = mat.cols;
        m_size.y = mat.rows;
        m_size.z = 1;
        m_p_void = mat.data;
    }
};

// Function prototypes
void cu_run_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img);
void cu_insert_image(CudaImg& t_cuda_big_img, CudaImg& t_cuda_small_img, int2 t_pos, uchar3 t_mask, bool t_is_and_mask);
void cu_insertimage( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position );
void cu_insert_mask(CudaImg& t_cuda_big_img, CudaImg& t_cuda_small_img, int2 t_pos, uchar3 t_mask, bool t_is_and_mask);
void cu_swap_image(CudaImg& img1, CudaImg& img2, int2 pos);
void cu_rotate90_rgba(CudaImg& src, CudaImg& dst, int direction);
void cu_rotate90(CudaImg& t_cu_img, CudaImg& t_cu_img_rotated, int t_direction);
void cu_scale(CudaImg& t_cu_img, CudaImg& t_cu_img_scaled);
void cu_draw_char(CudaImg& t_cu_img, int t_x, int t_y, char t_char, uchar3 t_color);
void initialize_font();
