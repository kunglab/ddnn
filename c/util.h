#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define BIT8 0x01
#define BIT7 0x02
#define BIT6 0x04
#define BIT5 0x08
#define BIT4 0x10
#define BIT3 0x20
#define BIT2 0x40
#define BIT1 0x80
#define BIT8 0x01
const char bits[8] = {BIT1, BIT2, BIT3, BIT4, BIT5, BIT6, BIT7, BIT8};
#define MAX_FLT_SIZE 12


/* indexing functions */
int idx_2d(int i, int j, int rows);
int idx_3d(int i, int j, int k, int rows, int cols);
int idx_4d(int i, int j, int k, int l, int rows, int cols, int depth);
int conv_idx(int pl_i, int x, int kx, int sx, int px);
int convpool_size(int x, int kx, int sx, int px, int pl_x, int pl_sx, int pl_px);
static int bslice_2d(uint8_t* dst, uint8_t* src, int x, int y,
              int w, int h, int kw, int kh);
int bslice_2d_filter(uint8_t* dst, uint8_t* src, int x, int y,
                     int w, int h, int kw, int kh);
int bslice_4d(uint8_t* dst, uint8_t* src, const int x, const int y, const int zi, const int zj,
              const int w, const int h, const int d, const int kw, const int kh);

/* layer types */
void blinear_layer(uint8_t* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                   float* Beta, float* Mean, float* Std, int m, int n, int k);
void fconv_layer(float* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                 float* Beta, float* Mean, float* Std, int m, int num_f,
                 int w, int h, int d, int kw, int kh, int sw, int sh,
                 int pw, int ph, int pl_w, int pl_h, int pl_sw, int pl_hw,
                 int pl_pw, int pl_ph);
void bconv_layer(uint8_t* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                 float* Beta, float* Mean, float* Std, int m, int num_f,
                 int w, int h, int d, int kw, int kh, int sw, int sh,
                 int pw, int ph, int pl_w, int pl_h, int pl_sw, int pl_hw,
                 int pl_pw, int pl_ph);

/* layer helper functions */
float BN(float f, float Gamma, float Beta, float Mean, float Std);
int bdot(uint8_t* A, uint8_t* B, int N);
void fconv(float* A, uint8_t* F, uint8_t* C, int c_start_idx, float Bias,
           float Gamma, float Beta, float Mean, float Std,
           int w, int h, int d, int kw, int kh, int sw, int sh,
           int pw, int ph, int pl_w, int pl_h, int pl_sw, int pl_sh,
           int pl_pw, int pl_ph);
void bconv(uint8_t* A, uint8_t* F, uint8_t* C, int c_start_idx, int z,
           float Bias, float Gamma, float Beta, float Mean, float Std,
           int w, int h, int d, int kw, int kh, int sw, int sh, int pw, int ph,
           int pl_w, int pl_h, int pl_sw, int pl_sh, int pl_pw, int pl_ph);
int bdot_3d(uint8_t* A, uint8_t* B, int x, int y, int z, int w, int h,
            int d, int kw, int kh);
float fdot_3d(float* A, uint8_t* B, int x, int y, int w, int h, int d,
            int kw, int kh);

/* Bit functions */
uint8_t rotr8 (uint8_t n, unsigned int c);
uint32_t rotr1 (uint32_t x);
int popcnt(uint32_t v);
int popcnt8(uint8_t v);
int nthbitset(uint8_t num, int bit);
int nthbitset_arr(uint8_t *num, const int bit);


/* index functions */
int idx_2d(int i, int j, int rows)
{
    return i * rows + j;
}

int idx_3d(int i, int j, int k, int rows, int cols)
{
    return i * rows * cols + j * cols + k;
}

int idx_4d(int i, int j, int k, int l, int rows, int cols, int depth)
{
  return i * rows * cols * depth + j * cols * depth + k * depth + l;
}

int conv_idx(int pl_i, int x, int kx, int sx, int px)
{
  int conv_sz = (x - kx + 2*px)/sx;
  return (pl_i < 0 || pl_i > conv_sz) ? -INT_MAX : pl_i * sx - px;
}

int convpool_size(int x, int kx, int sx, int px, int pl_x, int pl_sx, int pl_px)
{
  return ((x - kx + 2*px)/sx - pl_x + (2*pl_px) + 1)/pl_sx + 1;
}

/* 2d slice function on binary matrix (bit packed) */
static int bslice_2d(uint8_t* dst, uint8_t* src, int x, int y,
                     int w, int h, int kw, int kh)
{
  int i, j, n, idx, shift, bytes;
  uint8_t mask, bitset;

  /* initiaize dst */
  bytes = (kw*kh/8)+1;
  for (i = 0; i < bytes; ++i) {
    dst[i] = 0;
  }

  idx = 0;
  shift = 7;
  n = 0;
  for (i = x; i < x + kw; ++i) {
    for (j = y; j < y + kh; ++j) {
      /* Padding out of bounds */
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        continue;
      }

      bitset = nthbitset_arr(src, idx_2d(i, j, w));
      dst[idx/8] |= bitset << shift;

      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
      n++;
    }
  }

  return n;
}

int bslice_2d_filter(uint8_t* dst, uint8_t* src, int x, int y,
                     int w, int h, int kw, int kh)
{
  int i, j, n, idx, shift, bytes, bitset;
  uint8_t mask;

  /* initiaize dst */
  bytes = (kw*kh/8)+1;
  for (i = 0; i < bytes; ++i) {
    dst[i] = 0xFF;
  }

  idx = 0;
  shift = 7;
  n = 0;
  for (i = x; i < x + kw; ++i) {
    for (j = y; j < y + kh; ++j) {
      /* Padding out of bounds */
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        continue;
      }

      bitset = nthbitset_arr(src, idx_2d(i, j, w));
      dst[idx/8] &= ~((!bitset) << shift);

      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
      n++;
    }
  }

  return n;
}

/* 4d slice function on binary matrix (bit packed) */
int bslice_4d(uint8_t* dst, uint8_t* src, const int x, const int y, const int zi, const int zj,
              const int w, const int h, const int d, const int kw, const int kh)
{
  int i, j, n, idx, shift, bytes, bitset;
  uint8_t mask;

  /* initialize dest */
  bytes = (kw*kh/8)+1;
  for (i = 0; i < bytes; ++i) {
    dst[i] = 0;
  }

  idx = 0;
  shift = 7;
  n = 0;
  for (i = x; i < x + kw; ++i) {
    for (j = y; j < y + kh; ++j) {
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        continue;
      }

      bitset = nthbitset_arr(src, idx_4d(zi, zj, i, j, d, w, h));
      dst[idx/8] |= bitset << shift;
      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
      n++;
    }
  }

  return n;
}

/* Layers */
void blinear_layer(uint8_t* A, uint8_t* B, uint8_t* C, float* Bias, float* Gamma,
                   float* Beta, float* Mean, float* Std, int m, int n, int k)
{
  int i, j, ni, ri, ci,  max_idx;
  float res, max_res;

  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 7) / 8;
  for (i = 0; i < m; ++i) {
    max_res = -FLT_MAX;
    ri = i * ni;
    for (j = 0; j < k; ++j) {
      ci = j * ni;
      res = bdot(A + ri, B + ci, n);
      res += Bias[j];
      res = BN(res, Gamma[j], Beta[j], Mean[j], Std[j]);
      if (res > max_res) {
        max_idx = j;
        max_res = res;
      }
    }
    C[i] = max_idx;
  }
}

void fconv_layer(float* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                 float* Beta, float* Mean, float* Std, int m, int num_f,
                 int w, int h, int d, int kw, int kh, int sw, int sh,
                 int pw, int ph, int pl_w, int pl_h, int pl_sw, int pl_sh,
                 int pl_pw, int pl_ph)
{
  int i, j, res_size, res_w, res_h, c_idx, a_idx, f_idx;

  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      a_idx = i*w*h*d;
      f_idx = j*(((kw*kh*d)/8)+1);
      fconv(A + a_idx, F + f_idx, C, c_idx, Bias[j], Gamma[j],
            Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}

void bconv_layer(uint8_t* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                 float* Beta, float* Mean, float* Std, int m, int num_f,
                 int w, int h, int d, int kw, int kh, int sw, int sh,
                 int pw, int ph, int pl_w, int pl_h, int pl_sw, int pl_sh,
                 int pl_pw, int pl_ph)
{
  int i, j, res_size, res_w, res_h, c_idx, f_idx;

  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      f_idx = j*d*(((kw*kh)/8)+1);
      bconv(A, F + f_idx, C, c_idx, i, Bias[j], Gamma[j],
            Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}

/* layer helper functions */
float BN(float f, float Gamma, float Beta, float Mean, float Std)
{
  f -= Mean;
  f /= Std;
  f *= Gamma;
  f += Beta;
  return f;
}

int bdot(uint8_t* A, uint8_t* B, int N)
{
  int i, num_bytes, res;

  num_bytes = (N/8) + 1;
  res = 0;
  for (i = 0; i < num_bytes; ++i) {
    res += popcnt8(~(A[i]^B[i]));
  }
  res = res*2 - N;
  return res;
}

int bdot_3d(uint8_t* A, uint8_t* B, int x, int y, int z, int w, int h,
            int d, int kw, int kh)
{
  /* Handles up to 10x10 filters */
  uint8_t A_slice[MAX_FLT_SIZE] = {0};
  uint8_t B_slice[MAX_FLT_SIZE] = {0};
  uint8_t *B_idx;
  int i, comp_n, res, N, B_bytes, bx, by, bw, bh;

  N = kw*kh;
  B_bytes = (kw*kh)/8 + 1;
  res = 0;
  for (i = 0; i < d; ++i) {
    B_idx = B + B_bytes*i;
    comp_n = bslice_4d(A_slice, A, x, y, z, i, w, h, d, kw, kh);
    //no padding
    if (comp_n == N) {
      res += bdot(A_slice, B_idx, N);
    }
    //padding
    else {
      bx = -MIN(0, x);
      by = -MIN(0, y);
      bw = MIN(kw, w - x);
      bh = MIN(kh, h - y);
      bslice_2d_filter(B_slice, B_idx, bx, by, kw, kh, bw, bh);
      res += bdot(A_slice, B_slice, comp_n);
    }
  }

  return res;
}

/* float convolution + BN */
/* C_start_idx is the starting index for storing the result */
void fconv(float* A, uint8_t* F, uint8_t* C, int c_start_idx, float Bias,
           float Gamma, float Beta, float Mean, float Std,
           int w, int h, int d, int kw, int kh, int sw, int sh,
           int pw, int ph, int pl_w, int pl_h, int pl_sw, int pl_sh,
           int pl_pw, int pl_ph)
{
  uint8_t c_mask, res_sign;
  int pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, c_shift, c_idx;
  float res, max_res;

  c_shift = 7 - (c_start_idx % 8);
  c_mask = 1 << c_shift;
  c_idx = c_start_idx / 8;
  pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
  pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
  for (pl_i = -pl_pw; pl_i + pl_w + pl_pw - 1 < pl_i_max; pl_i += pl_sw) {
  for (pl_j = -pl_ph; pl_j + pl_h + pl_pw - 1 < pl_j_max; pl_j += pl_sh) {
    max_res = res = -FLT_MAX;
    for (i_in = pl_i; i_in < pl_i + pl_w; ++i_in) {
    i = conv_idx(i_in, w, kw, sw, pw);
    for (j_in = pl_j; j_in < pl_j + pl_h; ++j_in) {
      j = conv_idx(j_in, h, kh, sh, ph);
      if (i >= -pw && j >= -ph) {
        res = fdot_3d(A, F, i, j, w, h, d, kw, kh);
        max_res = MAX(res, max_res);
      }
    }
    }
    max_res += Bias;
    max_res = BN(max_res, Gamma, Beta, Mean, Std);
    res_sign = max_res >= 0 ? 1 : 0;

    /* store result */
    C[c_idx] |= res_sign << c_shift;

    /* update c_idx */
    c_mask = rotr1(c_mask);
    c_idx += (c_mask & 0x80) >> 7;
    c_shift--;
    c_shift =  c_shift < 0 ? 7 : c_shift;
  }
  }
}

void bconv(uint8_t* A, uint8_t* F, uint8_t* C, int c_start_idx, int z,
           float Bias, float Gamma, float Beta, float Mean, float Std,
           int w, int h, int d, int kw, int kh, int sw, int sh, int pw, int ph,
           int pl_w, int pl_h, int pl_sw, int pl_sh, int pl_pw, int pl_ph)
{
  uint8_t c_mask, res_sign;
  int pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, c_shift, c_idx;
  float res, max_res;

  c_shift = 7 - (c_start_idx % 8);
  c_mask = 1 << c_shift;
  c_idx = c_start_idx / 8;
  pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
  pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
  for (pl_i = -pl_pw; pl_i + pl_w + pl_pw - 1 < pl_i_max; pl_i += pl_sw) {
  for (pl_j = -pl_ph; pl_j + pl_h + pl_pw - 1 < pl_j_max; pl_j += pl_sh) {
    max_res = res = -FLT_MAX;
    for (i_in = pl_i; i_in < pl_i + pl_w; ++i_in) {
    i = conv_idx(i_in, w, kw, sw, pw);
    for (j_in = pl_j; j_in < pl_j + pl_h; ++j_in) {
      j = conv_idx(j_in, h, kh, sh, ph);
      if (i >= -pw && j >= -ph) {
        res = bdot_3d(A, F, i, j, z, w, h, d, kw, kh);
        max_res = MAX(res, max_res);
      }
    }
    }
    max_res += Bias;
    max_res = BN(max_res, Gamma, Beta, Mean, Std);
    res_sign = max_res >= 0 ? 1 : 0;

    /* store result */
    C[c_idx] |= res_sign << c_shift;

    /* update c_idx */
    c_mask = rotr1(c_mask);
    c_idx += (c_mask & 0x80) >> 7;
    c_shift--;
    c_shift =  c_shift < 0 ? 7 : c_shift;
  }
  }
}

float fdot_3d(float* A, uint8_t* B, int x, int y, int w, int h, int d,
              int kw, int kh)
{
  uint8_t  bitset;
  int i, j, k, b_idx, A_bytes;
  float *A_slice, a, res;

  A_bytes = w*h;
  res = 0;
  b_idx = 0;
  for (i = 0; i < d; ++i) {
    A_slice = A + A_bytes*i;
    for (j = x; j < x + kw; ++j) {
      for (k = y; k < y + kh; ++k) {
        /* handles padding */
        if (j < 0 || j > h-1 || k < 0 || k > w-1) {
          a = 0.0;
        }
        else {
          a = A_slice[idx_2d(j, k, w)];
        }

        bitset = nthbitset_arr(B, b_idx);
        res += bitset ? a : -a;
        b_idx++;
      }
    }
  }

  return res;
}




/* Bit functions */
uint8_t rotr8 (uint8_t n, unsigned int c)
{
  const unsigned int mask = (CHAR_BIT*sizeof(n)-1);
  return (n>>c) | (n<<( (-c)&mask ));
}

uint32_t rotr1 (uint32_t x)
{
  return (x>>1) | (x<<(7));
}

int nthbitset(uint8_t x, int n)
{
  return x & (1 << n) ? 1 : 0;
}

int nthbitset_arr(uint8_t *arr, const int n)
{
  return arr[n/8] & bits[n&7] ? 1 : 0;
}

int popcnt(uint32_t v) {
  uint32_t c = 0;
  c = v - ((v >> 1) & 0x55555555);
  c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
  c = ((c >> 4) + c) & 0x0F0F0F0F;
  c = ((c >> 8) + c) & 0x00FF00FF;
  return ((c >> 16) + c) & 0x0000FFFF;
}

int popcnt8(uint8_t v) {
  uint8_t c = 0;
  c = v - ((v >> 1) & 0x55);
  c = ((c >> 2) & 0x33) + (c & 0x33);
  return ((c >> 4) + c) & 0x0F;
}

#endif /*UTIL_H*/
