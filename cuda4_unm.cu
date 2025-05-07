// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
//
// Example of CUDA Technology Usage with unified memory.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__constant__ uint8_t d_font8x8[256][8]; // Konstantní paměť pro font


// === GRAYSCALE ===
__global__ void kernel_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y || l_x >= t_color_cuda_img.m_size.x) return;

    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];
    t_bw_cuda_img.m_p_uchar1[l_y * t_bw_cuda_img.m_size.x + l_x].x =
        l_bgr.x * 0.11f + l_bgr.y * 0.59f + l_bgr.z * 0.30f;
}

void cu_run_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;
    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size,
                  (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_grayscale<<<l_blocks, l_threads>>>(t_color_cuda_img, t_bw_cuda_img);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

// === INSERT IMAGE ===
__global__ void kernel_insert_image(CudaImg big_img, CudaImg small_img, int2 pos, uchar3 mask, bool is_and_mask)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_x >= small_img.m_size.x || l_y >= small_img.m_size.y) return;

    int big_x = pos.x + l_x;
    int big_y = pos.y + l_y;
    if (big_x >= big_img.m_size.x || big_y >= big_img.m_size.y) return;

    uchar3 sp = small_img.at3(l_y, l_x);
    uchar3& bp = big_img.at3(big_y, big_x);

    if (is_and_mask)
    {
        bp.x = sp.x & mask.x;
        bp.y = sp.y & mask.y;
        bp.z = sp.z & mask.z;
    }
    else
    {
        bp.x = sp.x * mask.x;
        bp.y = sp.y * mask.y;
        bp.z = sp.z * mask.z;
    }
}

void cu_insert_image(CudaImg& big_img, CudaImg& small_img, int2 pos, uchar3 mask, bool is_and_mask)
{
    dim3 block(16, 16);
    dim3 grid((small_img.m_size.x + block.x - 1) / block.x,
              (small_img.m_size.y + block.y - 1) / block.y);

    kernel_insert_image<<<grid, block>>>(big_img, small_img, pos, mask, is_and_mask);
    cudaDeviceSynchronize();
}

// === INSERT MASK (kopie insert_image) ===
__global__ void kernel_insert_mask(CudaImg big, CudaImg small, int2 pos, uchar3 mask, bool is_and_mask)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= small.m_size.x || y >= small.m_size.y) return;

    int big_x = pos.x + x;
    int big_y = pos.y + y;
    if (big_x >= big.m_size.x || big_y >= big.m_size.y) return;

    uchar3 sp = small.at3(y, x);
    uchar3& bp = big.at3(big_y, big_x);

    if (is_and_mask)
    {
        bp.x = sp.x & mask.x;
        bp.y = sp.y & mask.y;
        bp.z = sp.z & mask.z;
    }
    else
    {
        bp.x = sp.x * mask.x;
        bp.y = sp.y * mask.y;
        bp.z = sp.z * mask.z;
    }
}

void cu_insert_mask(CudaImg& big, CudaImg& small, int2 pos, uchar3 mask, bool is_and_mask)
{
    dim3 block(16, 16);
    dim3 grid((small.m_size.x + block.x - 1) / block.x,
              (small.m_size.y + block.y - 1) / block.y);

    kernel_insert_mask<<<grid, block>>>(big, small, pos, mask, is_and_mask);
    cudaDeviceSynchronize();
}

// === SWAP IMAGE ===
__global__ void kernel_swap_image(CudaImg big_img, CudaImg small_img, int2 pos)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= small_img.m_size.x || y >= small_img.m_size.y) return;

    int big_x = pos.x + x;
    int big_y = pos.y + y;
    if (big_x >= big_img.m_size.x || big_y >= big_img.m_size.y) return;

    uchar3 tmp = big_img.at3(big_y, big_x);
    big_img.at3(big_y, big_x) = small_img.at3(y, x);
    small_img.at3(y, x) = tmp;
}

void cu_swap_image(CudaImg& big_img, CudaImg& small_img, int2 pos)
{
    dim3 block(16, 16);
    dim3 grid((small_img.m_size.x + block.x - 1) / block.x,
              (small_img.m_size.y + block.y - 1) / block.y);

    kernel_swap_image<<<grid, block>>>(big_img, small_img, pos);
    cudaDeviceSynchronize();
}

__global__ void kernel_rotate90(CudaImg src, CudaImg dst, int direction) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    int new_x, new_y;

    if (direction == 1) { // Rotate 90 degrees clockwise
        new_x = src.m_size.y - 1 - y;
        new_y = x;
    } else if (direction == -1) { // Rotate 90 degrees counterclockwise
        new_x = y;
        new_y = src.m_size.x - 1 - x;
    } else {
        return; // Invalid direction
    }

    dst.at3(new_y, new_x) = src.at3(y, x);
}

void cu_rotate90(CudaImg& t_cu_img, CudaImg& t_cu_img_rotated, int t_direction) {
    dim3 block_size(16, 16);
    dim3 grid_size((t_cu_img.m_size.x + block_size.x - 1) / block_size.x,
                   (t_cu_img.m_size.y + block_size.y - 1) / block_size.y);

    kernel_rotate90<<<grid_size, block_size>>>(t_cu_img, t_cu_img_rotated, t_direction);
    cudaDeviceSynchronize();
}

__global__ void kernel_scale(CudaImg src, CudaImg dst, float scale_x, float scale_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.m_size.x || y >= dst.m_size.y) return;

    // Calculate the corresponding position in the source image
    float orig_x = x * scale_x;
    float orig_y = y * scale_y;

    int int_orig_x = (int)orig_x;
    int int_orig_y = (int)orig_y;

    float diff_x = orig_x - int_orig_x;
    float diff_y = orig_y - int_orig_y;

    // Boundary check for bilinear interpolation
    if (int_orig_x + 1 >= src.m_size.x || int_orig_y + 1 >= src.m_size.y) return;

    uchar3 bgr00 = src.at3(int_orig_y, int_orig_x);
    uchar3 bgr01 = src.at3(int_orig_y, int_orig_x + 1);
    uchar3 bgr10 = src.at3(int_orig_y + 1, int_orig_x);
    uchar3 bgr11 = src.at3(int_orig_y + 1, int_orig_x + 1);

    uchar3 bgr;
    bgr.x = bgr00.x * (1 - diff_y) * (1 - diff_x) +
            bgr01.x * (1 - diff_y) * diff_x +
            bgr10.x * diff_y * (1 - diff_x) +
            bgr11.x * diff_y * diff_x;

    bgr.y = bgr00.y * (1 - diff_y) * (1 - diff_x) +
            bgr01.y * (1 - diff_y) * diff_x +
            bgr10.y * diff_y * (1 - diff_x) +
            bgr11.y * diff_y * diff_x;

    bgr.z = bgr00.z * (1 - diff_y) * (1 - diff_x) +
            bgr01.z * (1 - diff_y) * diff_x +
            bgr10.z * diff_y * (1 - diff_x) +
            bgr11.z * diff_y * diff_x;

    dst.at3(y, x) = bgr;
}

void cu_scale(CudaImg& t_cu_orig, CudaImg& t_cu_scaled) {
    float scale_x = (float)(t_cu_orig.m_size.x - 1) / t_cu_scaled.m_size.x;
    float scale_y = (float)(t_cu_orig.m_size.y - 1) / t_cu_scaled.m_size.y;

    dim3 block_size(16, 16);
    dim3 grid_size((t_cu_scaled.m_size.x + block_size.x - 1) / block_size.x,
                   (t_cu_scaled.m_size.y + block_size.y - 1) / block_size.y);

    kernel_scale<<<grid_size, block_size>>>(t_cu_orig, t_cu_scaled, scale_x, scale_y);
    cudaDeviceSynchronize();
}

__global__ void kernel_draw_char(CudaImg img, int x, int y, char c, uchar3 color) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Zkontrolujeme, zda je vlákno uvnitř znaku
    if (tx >= 8 || ty >= 8) return;

    // Získání bitu z fontu
    uint8_t font_row = d_font8x8[(int)c][ty];
    if (font_row & (1 << (7 - tx))) { // Pokud je bit nastaven
        int px = x + tx;
        int py = y + ty;

        // Zkontrolujeme, zda je pixel uvnitř obrázku
        if (px >= 0 && px < img.m_size.x && py >= 0 && py < img.m_size.y) {
            img.at3(py, px) = color; // Nastavení barvy pixelu
        }
    }
}

void cu_draw_char(CudaImg& img, int x, int y, char c, uchar3 color) {
    dim3 block_size(8, 8); // Velikost bloku odpovídá velikosti znaku (8x8)
    dim3 grid_size(1, 1);  // Jeden blok pro jeden znak

    kernel_draw_char<<<grid_size, block_size>>>(img, x, y, c, color);
    cudaDeviceSynchronize();
}

void initialize_font() {
    cudaMemcpyToSymbol(d_font8x8, font8x8, sizeof(font8x8));
}