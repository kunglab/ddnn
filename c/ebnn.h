#ifndef EBNN_H
#define EBNN_H

#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define CEIL_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define MAX_FILTER_BYTES 12
static const uint8_t bits[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};


/* layer types */
static void blinear_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int m, const int n,
                          const int k);
static void fconv_layer(const float* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);
static void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);

/* layer helper functions */
static float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std);
static void fconv(const float* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph);
static void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const int z, const float Bias,
                  const float Gamma, const float Beta, const float Mean,
                  const float Std, const int w, const int h, const int d,
                  const int kw, const int kh, const int sw, const int sh,
                  const int pw, const int ph, const int pl_w, const int pl_h,
                  const int pl_sw, const int pl_sh, const int pl_pw,
                  const int pl_ph);
static float fdot_3d(const float* A, const uint8_t* B, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh);
static int bdot_3d(const uint8_t* A, const uint8_t* B, const int x, const int y,
                   const int z, const int w, const int h, const int d,
                   const int kw, const int kh);
static int bdot(const uint8_t* A, const uint8_t* B, const int N);

/* indexing functions */
static int idx_2d(const int i, const int j, const int rows);
static int idx_3d(const int i, const int j, const int k, const int rows,
                  const int cols);
static int idx_4d(const int i, const int j, const int k, const int l,
                  const int rows, const int cols, const int depth);
static int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px);
static int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px);

/* Bit functions */
static uint8_t rotr1 (const uint8_t x);
static int popcnt8(const uint8_t v);
static int nthbitset_arr(const uint8_t* const arr, const int n);
static int bslice_2d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int w, const int h, const int kw,
                     const int kh);
static int bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int x, const int y, const int w, const int h,
                            const int kw, const int kh);
static int bslice_4d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int zi, const int zj, const int w,
                     const int h, const int d, const int kw, const int kh);



/* layers types */
static void blinear_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int m, const int n,
                          const int k)
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
      res = bdot(A + ri, F + ci, n);
      res += Bias[j];
      res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]);
      if (res > max_res) {
        max_idx = j;
        max_res = res;
      }
    }
    C[i] = max_idx;
  }
}
static void fconv_layer(const float* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph)
{
  int i, j, max_m, res_size, res_w, res_h, c_idx, a_idx, f_idx;

  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;
  max_m = CEIL_POS(res_size*m*num_f/8.0);

  /* initialize result */
  for (i = 0; i < max_m; ++i) {
    C[i] = 0;
  }

  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      a_idx = i*w*h*d;
      f_idx = j*CEIL_POS(kw*kh*d/8.0);
      fconv(A + a_idx, F + f_idx, C, c_idx, Bias[j], Gamma[j],
            Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}

static void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph)
{
  int i, j, max_m, res_size, res_w, res_h, c_idx, f_idx;


  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;
  max_m = CEIL_POS(res_size*m*num_f/8.0);

  /* initialize result */
  for (i = 0; i < max_m; ++i) {
    C[i] = 0;
  }

  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      f_idx = j*d*CEIL_POS(kw*kh/8.0);
      bconv(A, F + f_idx, C, c_idx, i, Bias[j], Gamma[j],
            Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}

/* layer helper functions */
static float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std)
{
  f -= Mean;
  f /= Std;
  f *= Gamma;
  f += Beta;
  return f;
}

static int bdot(const uint8_t* A, const uint8_t* B, const int N)
{
  int i, num_bytes, res;

  num_bytes = CEIL_POS(N/8.0);
  res = 0;
  for (i = 0; i < num_bytes; ++i) {
    res += popcnt8(~(A[i]^B[i]));
  }
  res = res*2 - N;
  return res;
}

static int bdot_3d(const uint8_t* A, const uint8_t* B, const int x, const int y,
                    const int z, const int w, const int h, const int d,
                    const int kw, const int kh)
{
  /* Handles up to 10x10 filters */
  uint8_t A_slice[MAX_FILTER_BYTES] = {0};
  uint8_t B_slice[MAX_FILTER_BYTES] = {0};
  const uint8_t *B_idx;
  int i, comp_n, res, N, B_bytes, bx, by, bw, bh;

  N = kw*kh;
  B_bytes = CEIL_POS(kw*kh/8.0);
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
static void fconv(const float* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph)
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
    max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);
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

static void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const int z, const float Bias,
                  const float Gamma, const float Beta, const float Mean,
                  const float Std, const int w, const int h, const int d,
                  const int kw, const int kh, const int sw, const int sh,
                  const int pw, const int ph, const int pl_w, const int pl_h,
                  const int pl_sw, const int pl_sh, const int pl_pw,
                  const int pl_ph)
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
    max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);
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

static float fdot_3d(const float* A, const uint8_t* B, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh)
{
  uint8_t  bitset;
  int i, j, k, b_idx, A_bytes;
  float a, res;
  const float *A_slice;

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

/* index functions */
static int idx_2d(const int i, const int j, const int rows)
{
  return i * rows + j;
}

static int idx_3d(const int i, const int j, const int k, const int rows,
                  const int cols)
{
  return i * rows * cols + j * cols + k;
}

static int idx_4d(const int i, const int j, const int k, const int l,
                  const int rows, const int cols, const int depth)
{
  return i * rows * cols * depth + j * cols * depth + k * depth + l;
}

static int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px)
{
  int conv_sz = (x - kx + 2*px)/sx;
  return (pl_i < 0 || pl_i > conv_sz) ? -INT_MAX : pl_i * sx - px;
}

static int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px)
{
  return ((x - kx + 2*px)/sx - pl_x + (2*pl_px) + 1)/pl_sx + 1;
}

/* Bit functions */
static uint8_t rotr1 (const uint8_t x)
{
  return (x >> 1) | (x << 7);
}

static int nthbitset_arr(const uint8_t* const arr, const int n)
{
  return arr[n/8] & bits[n%8] ? 1 : 0;
}

static int popcnt8(const uint8_t v) {
  uint8_t c;
  c = v - ((v >> 1) & 0x55);
  c = ((c >> 2) & 0x33) + (c & 0x33);
  return ((c >> 4) + c) & 0x0F;
}

static int bslice_2d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int w, const int h, const int kw,
                     const int kh)
{
  int i, j, n, idx, shift, bytes;
  uint8_t mask, bitset;

  /* initiaize dst */
  bytes = CEIL_POS(kw*kh/8.0);
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

static int bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int x, const int y, const int w, const int h,
                            const int kw, const int kh)
{
  int i, j, n, idx, shift, bytes, bitset;
  uint8_t mask;

  /* initiaize dst */
  bytes = CEIL_POS(kw*kh/8.0);
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
static int bslice_4d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int zi, const int zj, const int w,
                     const int h, const int d, const int kw, const int kh)
{
  int i, j, n, idx, shift, bytes, bitset;
  uint8_t mask;

  /* initialize dest */
  bytes = CEIL_POS(kw*kh/8.0);
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

#endif /*EBNN_H*/
