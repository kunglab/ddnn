#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define MAX_FLT_SIZE 12


/* indexing functions */
int idx_2d(int i, int j, int rows);
int idx_3d(int i, int j, int k, int rows, int cols);
int idx_4d(int i, int j, int k, int l, int rows, int cols, int depth);

void bslice_2d(uint8_t* dst, uint8_t* src, int x, int y,
               int w, int h, int kw, int kh);

/* layer types */
void fused_float_linear_layer(float* A, uint8_t* W, uint8_t* C,
                              float* Bias, float* Gamma, float* Beta,
                              float* Mean, float* Std, int m, int n, int k);
void fused_linear_layer(uint8_t* A, uint8_t* B, uint8_t* C,
                        float* Bias, float* Gamma, float* Beta,
                        float* Mean, float* Std, int m, int n, int k);
void fused_float_conv_layer(float* A, uint8_t* F, uint8_t* C,
                            float* Bias, float* Gamma, float* Beta,
                            float* Mean, float* Std, int m,
                            int n, int w, int h, int num_f, int kw, int kh,
                            int sw, int sh, int pw, int ph, int flat_output);
void fconv_layer(float* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                 float* Beta, float* Mean, float* Std, int m, int num_f,
                 int w, int h, int d, int kw, int kh, int sw, int sh,
                 int pw, int ph);
void fused_float_conv_pool_layer(float* A, uint8_t* F, uint8_t* C,
                                 float* Bias, float* Gamma, float* Beta,
                                 float* Mean, float* Std, int m,
                                 int n, int w, int h, int num_f, int kw, int kh,
                                 int sw, int sh, int pw, int ph, int ps);
void fused_conv_layer(uint8_t* A, uint8_t* F, uint8_t* C,
                      float* Bias, float* Gamma, float* Beta,
                      float* Mean, float* Std, int m,
                      int n, int w, int h, int num_f, int kw, int kh,
                      int sw, int sh, int pw, int ph, int flat_output);
void fused_conv_pool_layer(uint8_t* A, uint8_t* F, uint8_t* C,
                           float* Bias, float* Gamma, float* Beta,
                           float* Mean, float* Std, int m,
                           int n, int w, int h, int num_f, int kw, int kh,
                           int sw, int sh, int pw, int ph, int ps);
float BN(float f, float Gamma, float Beta, float Mean, float Std);
void linear_softmax_layer(uint8_t* A, uint8_t* B, uint8_t* C, float* Bias,
                          int m, int n, int k);
void linear_BN_softmax_layer(uint8_t* A, uint8_t* B, uint8_t* C,
                             float* Bias, float* Gamma, float* Beta,
                             float* Mean, float* Std, int m, int n, int k);

/* Convolution functions */
float dot3(uint8_t* A, uint8_t* W, int num_chan, int i, int j, int w, int h);
float bconv(uint8_t* A, uint8_t* W, int num_chan, int i, int j, int w, int h,
           int kw, int kh);
int bdot(uint8_t* A, uint8_t* B, int N);
int bdot_3d(uint8_t* A, uint8_t* B, int x, int y, int w, int h, int d,
            int kw, int kh);
float fdot_3d(float* A, uint8_t* B, int x, int y, int w, int h, int d,
            int kw, int kh);
int fconv(float* A, uint8_t* F, uint8_t* C, int c_start_idx, float Bias,
          float Gamma, float Beta, float Mean, float Std,
          int w, int h, int d, int kw, int kh, int sw, int sh,
          int pw, int ph);

/* Bit functions */
uint8_t rotr8 (uint8_t n, unsigned int c);
uint32_t rotr1 (uint32_t x);
int popcnt(uint32_t v);
int popcnt8(uint8_t v);
int nthbitset(uint8_t num, int bit);
int nthbitset_arr(uint8_t *num, int bit);
void printbits(uint8_t v, int n);
void print_binary_mat(uint8_t *a, int M, int N, int row_major);
void print_float_mat(float *a, int M, int N);


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


/* 2d slice function on binary matrix (bit packed) */
void bslice_2d(uint8_t* dst, uint8_t* src, int x, int y,
               int w, int h, int kw, int kh)
{
  int i, j, idx, shift;
  uint8_t mask, bitset;

  /* initialize dest */
  for (i = 0; i < kw*kh; ++i) {
    dst[i] = 0;
  }

  idx = 0;
  shift = 7;
  for (i = x; i < x + kw; ++i) {
    for (j = y; j < y + kh; ++j) {
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        bitset = 0;
      }
      else {
        bitset = nthbitset_arr(src, idx_2d(i, j, w));
      }

      dst[idx/8] |= bitset << shift;

      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
    }
  }
}


/* Layers */
void fused_float_linear_layer(float* A, uint8_t* W, uint8_t* C,
                              float* Bias, float* Gamma, float* Beta,
                              float* Mean, float* Std, int m, int n, int k)
{
  int i, row, col, ni, ki, a_idx, w_idx, w_val, c_idx, res_sign;
  int c_shift;
  uint8_t w_mask, c_mask;
  float a_val, res;

  ni = (n + 8 - 1) / 8;
  ki = (k + 8 - 1) / 8;
  c_idx = 0;

  /* Initalize the result matrix */
  for (i = 0; i < m*ki; ++i) C[i] = 0;

  for (row = 0; row < m; ++row) {
    c_shift = 7;
    c_mask = 0x80;

    for (col = 0; col < k; ++col) {
      a_idx = row * n;
      w_idx = col * ni;
      w_mask = 0x80;
      res = 0;

      for (i = 0; i < n; ++i) {
        a_val = A[a_idx];
        w_val = (W[w_idx] & w_mask);
        res += w_val > 0 ? a_val : -a_val;

        /* update matrix positions */
        a_idx++;
        w_mask = (w_mask >> 1 | w_mask << 7);
        w_idx += (w_mask & 0x80) >> 7;
      }

      res += Bias[col];
      res -= Mean[col];
      res /= Std[col];
      res *= Gamma[col];
      res += Beta[col];
      res_sign = res > 0 ? 1 : 0;

      //Need to shift to correct bit location
      C[c_idx] |= res_sign << c_shift;
      c_mask = rotr1(c_mask);
      c_idx += (c_mask & 0x80) >> 7;
      c_shift--;
      c_shift =  c_shift < 0 ? 7 : c_shift;
    }
  }
}

void fused_linear_layer(uint8_t* A, uint8_t* B, uint8_t* C,
                        float* Bias, float* Gamma, float* Beta,
                        float* Mean, float* Std, int m, int n, int k)
{
  int i, row, col, ni, ki, ri, ci, c_idx, res_sign, c_shift;
  uint8_t c_mask;
  float res;

  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 8 - 1) / 8;
  ki = (k + 8 - 1) / 8;

  /* Initalize the result matrix */
  for (i = 0; i < m*ki; ++i) C[i] = 0;

  c_idx = 0;
  for (row = 0; row < m; ++row) {
    c_shift = 7;
    c_mask = 0x80;
    for (col = 0; col < k; ++col) {
      ri = row * ni;
      ci = col * ni;
      res = 0;

      for (i = 0; i < ni; ++i) {
        res += popcnt8(~(A[ri + i]^B[ci + i]));
      }

      /* needed after popcount */
      res = res*2 - n;
      res += Bias[col];

      /* Batch Norm */
      res -= Mean[col];
      res /= Std[col];
      res *= Gamma[col];
      res += Beta[col];
      res_sign = res > 0 ? 1 : 0;

      //Need to shift to correct bit location
      C[c_idx] |= (res_sign << c_shift);
      c_mask = rotr1(c_mask);
      c_idx += (c_mask & 0x80) >> 7;
      c_shift--;
      c_shift = c_shift < 0 ? 7 : c_shift;
    }
  }
}

void fused_float_conv_layer(float* A, uint8_t* F, uint8_t* C,
                            float* Bias, float* Gamma, float* Beta,
                            float* Mean, float* Std, int m,
                            int n, int w, int h, int num_f, int kw, int kh,
                            int sw, int sh, int pw, int ph, int flat_output)
{
  int mi, i, j, f, c_shift, bin_f_len, res_w, res_h, c_idx, f_idx, a_idx;
  uint8_t res_sign, c_mask;
  float res;

  /* packed res_w stride */
  res_w = (w + (pw * 2) - kw + 1 + 7) / 8;
  res_h = h + (ph * 2) - kh + 1;
  bin_f_len = (kw * kh + 7) / 8;
  c_idx = 0;

  /* Initalize the result matrix */
  for (mi = 0; mi < m*num_f*res_w*res_h; ++mi) C[mi] = 0;

  c_shift = 7;
  c_mask = 0x80;
  for (mi = 0; mi < m; ++mi) {
    a_idx = idx_4d(mi, 0, 0, 0, n, w, h);
    for (f = 0; f < num_f; ++f) {
      f_idx = f * bin_f_len * n;
      for (i = -pw; i < w - kw + pw + 1; i += sw) {
        if (!flat_output) {
          c_shift = 7;
          c_mask = 0x80;
        }

        for (j = -ph; j < h - kh + ph + 1; j += sh) {
          /* compute conv and BN */
          res = fdot_3d(A + a_idx, F + f_idx, i, j, w, h, n, kw, kh);
          res += Bias[f];
          res = BN(res, Gamma[f], Beta[f], Mean[f], Std[f]);
          res_sign = res > 0 ? 1 : 0;

          /* store result */
          C[c_idx] |= res_sign << c_shift;

          /* update c_idx */
          c_mask = rotr1(c_mask);
          c_idx += (c_mask & 0x80) >> 7;
          c_shift--;
          c_shift =  c_shift < 0 ? 7 : c_shift;
        }

        /* aligns rows on byte */
        if (!flat_output && c_mask != 0x80) c_idx++;
      }
    }
  }
}

void fused_float_conv_pool_layer(float* A, uint8_t* F, uint8_t* C,
                                 float* Bias, float* Gamma, float* Beta,
                                 float* Mean, float* Std, int m,
                                 int n, int w, int h, int num_f, int kw, int kh,
                                 int sw, int sh, int pw, int ph, int ps)
{
  int mi, i, j, pi, pj, f, c_shift, bin_f_len,  c_idx,
      f_idx, a_idx, conv_w, conv_h, pool_w, pool_h, res_w, res_h,
      cv_i, cv_j;
  uint8_t res_sign, c_mask;
  float res, max_res;

  /* packed res_w stride */
  //conv_h = h/sh-kh/2;
  //conv_w = w/sw-kw/2;
  //pool_h = conv_h/ps-ph/2;
  //pool_w = conv_w/ps-pw/2;

  if(sh<kh){
      conv_h = (h-kh+sh)/sh;
  }else{
      conv_h = h/sh;
  }
  if(sw<kw){
      conv_w = (w-kw+sw)/sw;
  }else{
      conv_w = w/sw;
  }
  if(ps<ph){
      pool_h = (conv_h-ph+ps)/ps;
  }else{
      pool_h = (conv_h)/ps;
  }
  if(ps<pw){
      pool_w = (conv_w-pw+ps)/ps;
  }else{
      pool_w = (conv_w)/ps;
  }

  res_w =  (pool_w + 7) / 8;
  res_h = pool_h;
  bin_f_len = (kw * kh + 7) / 8;
  c_idx = 0;

  //printf("%d %d\n%d %d\n", conv_w, conv_h, pool_w, pool_h);
  /* printf("%d", ps); */
  /* Initalize the result matrix */
  for (mi = 0; mi < m*num_f*res_w*res_h; ++mi) C[mi] = 0;

  for (mi = 0; mi < m; ++mi) {
    a_idx = idx_4d(mi, 0, 0, 0, n, w, h);
    for (f = 0; f < num_f; ++f) {
      f_idx = f * bin_f_len * n;
      for (i = 0; i < pool_w; ++i) {
        c_shift = 7;
        c_mask = 0x80;
        for (j = 0; j < pool_h; ++j) {
          max_res = -FLT_MAX;
          cv_i = i*sw*ps;
          cv_j = j*sh*ps;

          //printf("PARA: %d %d %d %d\n", pw, ph, sw, sh);
          /* pooling window */
          for (pi = cv_i; pi < cv_i + pw*sw; pi += sw) {
            for (pj = cv_j; pj < cv_j + ph*sh; pj += sh) {
              /* compute conv and BN */
              res = fdot_3d(A + a_idx, F + f_idx, pi, pj, w, h, n, kw, kh);
              res += Bias[f];
              //printf("PARA: %d %d %d %d \n", w, h, kw, kh);
              //printf("CONV: %d/%d %d %d %f\n", f, num_f, pi, pj, res);

              max_res = MAX(res, max_res);
              /* printf("%d %d %f \n", pi, pj, res); */
            }
          }

          /*  */
          //printf("RES-bf: %d/%d %d %d %f\n", f, num_f, i, j, max_res);
          /*  */
          max_res = BN(max_res, Gamma[f], Beta[f], Mean[f], Std[f]);
          //printf("RES-at: %d/%d %d %d %f\n", f, num_f, i, j, max_res);
          res_sign = max_res > 0 ? 1 : 0;

          /* store result */
          C[c_idx] |= res_sign << c_shift;

          /* update idx */
          c_mask = rotr1(c_mask);
          c_idx += (c_mask & 0x80) >> 7;
          c_shift--;
          c_shift =  c_shift < 0 ? 7 : c_shift;
        }

        /* aligns rows on byte */
        if (c_mask != 0x80) c_idx++;
      }
    }
  }
}

void fused_conv_layer(uint8_t* A, uint8_t* F, uint8_t* C,
                      float* Bias, float* Gamma, float* Beta,
                      float* Mean, float* Std, int m,
                      int n, int w, int h, int num_f, int kw, int kh,
                      int sw, int sh, int pw, int ph, int flat_output)
{
  int mi, i, j, f, bin_f_len, bin_w, res_w, res_h, c_idx, f_idx, a_idx, c_shift;
  float res;
  uint8_t c_mask, res_sign;

  /* packed res_w stride */
  bin_w = (w + 7) / 8;
  res_w = (w + (pw * 2) - kw + 1 + 7) / 8;
  res_h = h + (ph * 2) - kh + 1;
  bin_f_len = (kw * kh + 7) / 8;

  /* Initalize the result matrix */
  for (mi = 0; mi < m*num_f*res_w*res_h; ++mi) C[mi] = 0;

  c_idx = 0;
  for (mi = 0; mi < m; ++mi) {
    a_idx = idx_4d(mi, 0, 0, 0, n, h, bin_w);
    for (f = 0; f < num_f; ++f) {
      f_idx = f * bin_f_len * n;
      for (i = -pw; i < w - kw + 1; i += sw) {
        if (!flat_output) {
          c_shift = 7;
          c_mask = 0x80;
        }

        for (j = -ph; j < h - kh + 1; j += sh) {
          /* compute conv and BN */
          res = bconv(A + a_idx, F + f_idx, n, i, j, w, h, kw, kh);
          res += Bias[f];
          res = BN(res, Gamma[f], Beta[f], Mean[f], Std[f]);
          res_sign = res > 0 ? 1 : 0;

          /* store result */
          C[c_idx] |= res_sign << c_shift;

          /* update idx */
          c_mask = rotr1(c_mask);
          c_idx += (c_mask & 0x80) >> 7;
          c_shift--;
          c_shift =  c_shift < 0 ? 7 : c_shift;
        }

        /* aligns rows on byte */
        if (!flat_output && c_mask != 0x80) c_idx++;
      }
    }
  }
}

void fused_conv_pool_layer(uint8_t* A, uint8_t* F, uint8_t* C,
                           float* Bias, float* Gamma, float* Beta,
                           float* Mean, float* Std, int m,
                           int n, int w, int h, int num_f, int kw, int kh,
                           int sw, int sh, int pw, int ph, int ps)
{
  int mi, i, j, pi, pj, f, c_shift, bin_f_len,  c_idx, bin_w,
      f_idx, a_idx, conv_w, conv_h, pool_w, pool_h, res_w, res_h,
      cv_i, cv_j;
  float res, max_res;
  uint8_t c_mask, res_sign;


  /* indexs */
  //conv_h = h/sh-kh+kh/2;
  //conv_w = w/sw-kw+kw/2;
  //pool_h = conv_h/ps-ph+ph/2;
  //pool_w = conv_w/ps-pw+pw/2;

  if(sh<kh){
      conv_h = (h-kh+sh)/sh;
  }else{
      conv_h = h/sh;
  }
  if(sw<kw){
      conv_w = (w-kw+sw)/sw;
  }else{
      conv_w = w/sw;
  }
  if(ps<ph){
      pool_h = (conv_h-ph+ps)/ps;
  }else{
      pool_h = (conv_h)/ps;
  }
  if(ps<pw){
      pool_w = (conv_w-pw+ps)/ps;
  }else{
      pool_w = (conv_w)/ps;
  }

  //conv_h = (h-2*(kh/2))/sh;
  //conv_w = (w-2*(kw/2))/sw;
  //pool_h = (conv_h-2*(ph/2))/ps;
  //pool_w = (conv_w-2*(pw/2))/ps;

  bin_w = (w + 7) / 8;
  res_w =  (pool_w + 7) / 8;
  res_h = pool_h;
  bin_f_len = (kw * kh + 7) / 8;
  //c_idx = 0;
  /* printf("%d %d\n%d %d\n%d %d\n", w, h, conv_w, conv_h, pool_w, pool_h); */

  /* Initalize the result matrix */
  for (mi = 0; mi < m*num_f*res_w*res_h; ++mi) C[mi] = 0;

  c_idx = 0;
  for (mi = 0; mi < m; ++mi) {
    a_idx = idx_4d(mi, 0, 0, 0, n, h, bin_w);
    for (f = 0; f < num_f; ++f) {
      f_idx = f * bin_f_len * n;
      for (i = 0; i < pool_w; ++i) {
        c_shift = 7;
        c_mask = 0x80;
        for (j = 0; j < pool_h; ++j) {
          max_res = -FLT_MAX;
          cv_i = i*sw*ps;
          cv_j = j*sh*ps;

          //printf("PARA: %d %d %d %d\n", pw, ph, sw, sh);
          /* pooling window */
          for (pi = cv_i; pi < cv_i + pw*sw; pi += sw) {
            for (pj = cv_j; pj < cv_j + ph*sh; pj += sh) {
              /* compute conv and BN */
              res = dot3(A + a_idx, F + f_idx, n, pi, pj, w, h);
              res += Bias[f];
              //printf("CONV %d %d %f n:%d w:%d h:%d\n", pi, pj, res, n, w, h);
              max_res = MAX(res, max_res);
              /*  */
            }
          }

          //printf("RES-bf: %d/%d %d %d %f\n", f, num_f, i, j, max_res);
          /*  */
          max_res = BN(max_res, Gamma[f], Beta[f], Mean[f], Std[f]);
          //printf("RES-at: %d/%d %d %d %f\n", f, num_f, i, j, max_res);
          res_sign = max_res > 0 ? 1 : 0;

          /* store result */
          C[c_idx] |= res_sign << c_shift;

          /* update idx */
          c_mask = rotr1(c_mask);
          c_idx += (c_mask & 0x80) >> 7;
          c_shift--;
          c_shift =  c_shift < 0 ? 7 : c_shift;
        }

        /* aligns rows on byte */
        if (c_mask != 0x80) c_idx++;
      }
    }
  }
}

float BN(float f, float Gamma, float Beta, float Mean, float Std)
{
  f -= Mean;
  f /= Std;
  f *= Gamma;
  f += Beta;
  return f;
}


/* Computes the dot product and does a rowwise max */
void linear_softmax_layer(uint8_t* A, uint8_t* B, uint8_t* C,
                          float* Bias, int m, int n, int k)
{
  int i, row, col, ni, ki, ri, ci, max_idx, c_idx;
  float res, max_res;

  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 8 - 1) / 8;
  ki = (k + 8 - 1) / 8;

  /* Initalize the result matrix */
  for (i = 0; i < m*ki; ++i) C[i] = 0;

  max_idx = 0;
  c_idx = 0;
  for (row = 0; row < m; ++row) {
    max_res = -FLT_MAX;
    for (col = 0; col < k; ++col) {
      res = 0;
      ri = row * ni;
      ci = col * ni;

      for (i = 0; i < ni; ++i) {
        res += popcnt8(~(A[ri + i]^B[ci + i]));
      }

      /* needed after popcount */
      res = res*2 - n;
      res += Bias[col];

      /* Compare with current max */
      if (res > max_res) {
        max_res = res;
        max_idx = col;
      }
    }
    C[c_idx] = max_idx;
    c_idx++;
  }
}


/* Computes the dot product and does a rowwise max */
void linear_BN_softmax_layer(uint8_t* A, uint8_t* B, uint8_t* C,
                             float* Bias, float* Gamma, float* Beta,
                             float* Mean, float* Std, int m, int n, int k)
{
  int i, row, col, ni, ki, ri, ci, max_idx, c_idx;
  float res, max_res;

  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 8 - 1) / 8;
  ki = (k + 8 - 1) / 8;

  /* Initalize the result matrix */
  for (i = 0; i < m*ki; ++i) C[i] = 0;

  max_idx = 0;
  c_idx = 0;
  for (row = 0; row < m; ++row) {
    max_res = -FLT_MAX;
    for (col = 0; col < k; ++col) {
      res = 0;
      ri = row * ni;
      ci = col * ni;

      for (i = 0; i < ni; ++i) {
        res += popcnt8(~(A[ri + i]^B[ci + i]));
      }

      /* needed after popcount */
      res = res*2 - n;

      res += Bias[col];

      /* Batch Norm */
      res -= Mean[col];
      res /= Std[col];
      res *= Gamma[col];
      res += Beta[col];
      printf("res-at: %f\n", res);

      /* Compare with current max */
      if (res > max_res) {
        max_res = res;
        max_idx = col;
      }
    }
    C[c_idx] = max_idx;
    c_idx++;
  }
}

/* Convolution functions */
float bconv(uint8_t* A, uint8_t* W, int num_chan, int i, int j, int w, int h,
          int kw, int kh)
{
  int p, q, bin_w, chan, a_idx, f_idx, bin_f_len;
  uint8_t f_val, f_mask, a_val;
  float res;

  bin_w = (w + 7) / 8;
  bin_f_len = (kw * kh + 7) / 8;
  res = 0;
  for (chan = 0; chan < num_chan; ++chan) {
    f_idx = chan * bin_f_len;
    f_mask = 0x80;
    for (p = i; p < i + kw; ++p) {
      for (q = j; q < j + kh; ++q) {
        a_idx = idx_3d(chan, p, q, h, bin_w);
        /* for padding */
        if (p < 0 || p > w - kw || q < 0 || q > h - kh) {
          a_val = 0;
        } else {
          a_val = nthbitset_arr(A, a_idx);
        }

        f_val = (W[f_idx] & f_mask);
        res += f_val == a_val ? 1 : 0;

        /* update matrix positions */
        a_idx++;
        f_mask = (f_mask >> 1 | f_mask << 7);
        f_idx += (f_mask & 0x80) >> 7;
      }
    }
  }

  return res;
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


/* Handles up to 10x10 filters */
int bdot_3d(uint8_t* A, uint8_t* B, int x, int y, int w, int h, int d,
            int kw, int kh)
{
  uint8_t tmp_slice[MAX_FLT_SIZE] = {0};
  uint8_t *A_slice, *B_slice;
  int i, j, res, N, N_bytes, A_bytes, B_bytes;

  N = kw*kh;
  A_bytes = (w*h)/8 + 1;
  B_bytes = (kw*kh)/8 + 1;
  N_bytes = N/8 + 1;
  res = 0;
  for (i = 0; i < d; ++i) {
    A_slice = A + A_bytes*i;
    B_slice = B + B_bytes*i;
    bslice_2d(tmp_slice, A_slice, x, y, w, h, kw, kh);
    res += bdot(tmp_slice, B_slice, N);
    for (j = 0; j < N_bytes; ++j) {
      tmp_slice[j] = 0;
    }
  }

  return res;
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

/* float convolution + BN */
/* C_start_idx is the starting index for storing the result */
/* The result is the output size (eventually will change to precompute) */
int fconv(float* A, uint8_t* F, uint8_t* C, int c_start_idx, float Bias,
           float Gamma, float Beta, float Mean, float Std,
           int w, int h, int d, int kw, int kh, int sw, int sh,
           int pw, int ph)
{
  uint8_t c_mask, res_sign;
  int i, j, c_shift, c_idx, res_size;
  float res;

  c_shift = 7 - (c_start_idx % 8);
  c_mask = 0 | (1 << c_shift);
  c_idx = c_start_idx / 8;
  res_size = 0;
  for (i = -pw; i < w - kw + pw + 1; i += sw) {
    for (j = -ph; j < h - kh + ph + 1; j += sh) {
      res = fdot_3d(A, F, i, j, w, h, d, kw, kh);
      res += Bias;
      res = BN(res, Gamma, Beta, Mean, Std);
      res_sign = res > 0 ? 1 : 0;

      /* store result */
      C[c_idx] |= res_sign << c_shift;

      /* update c_idx */
      c_mask = rotr1(c_mask);
      c_idx += (c_mask & 0x80) >> 7;
      c_shift--;
      c_shift =  c_shift < 0 ? 7 : c_shift;
      res_size++;
    }
  }

  return res_size;
}


void fconv_layer(float* A, uint8_t* F, uint8_t* C, float* Bias, float* Gamma,
                 float* Beta, float* Mean, float* Std, int m, int num_f,
                 int w, int h, int d, int kw, int kh, int sw, int sh,
                 int pw, int ph)
{
  int i, j, res_size, c_idx, a_idx, f_idx;

  c_idx = 0;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      a_idx = i*w*h*d;
      f_idx = j*(((kw*kh*d)/8)+1);
      res_size = fconv(A + a_idx, F + f_idx, C, c_idx, Bias[j], Gamma[j],
                       Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph);
      c_idx += res_size;
    }
  }
}


float dot3(uint8_t* A, uint8_t* W, int num_chan, int i, int j, int w, int h)
{
  int chan, bin_w, a_idx, a_mask_idx, f_idx, r1, r2, r3;
  float res;

  uint8_t a_val;
  const uint8_t a1_mask[8] = {224, 112, 56, 28, 14, 7, 3, 1};
  const uint8_t a2_mask[8] = {0, 0, 0, 0, 0, 0, 128, 192};
  const uint8_t a1_shift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  const uint8_t a2_shift[8] = {0, 0, 0, 0, 0, 0, 2, 1};
  const uint8_t a2_b2_shift[8] = {0, 0, 0, 0, 0, 0, 0, 1};

  /* number of bytes to store filter */
  const uint8_t bin_f_len = 2;

  bin_w = (w + 7) / 8;
  res = 0;
  a_mask_idx = j % 8;
  for (chan = 0; chan < num_chan; ++chan) {
    f_idx = chan * bin_f_len;
    a_idx = idx_3d(chan, i, j/8, h, bin_w);
    r1 = a_idx;
    r2 = a_idx + bin_w;
    r3 = a_idx + 2*bin_w;

    /* First byte */
    a_val = 0;
    a_val |= (A[r1] & a1_mask[a_mask_idx]) << a1_shift[a_mask_idx];
    a_val |= (A[r1+1] & a2_mask[a_mask_idx]) >> a2_shift[a_mask_idx];

    a_val |= ((A[r2] & a1_mask[a_mask_idx]) << a1_shift[a_mask_idx]) >> 3;
    a_val |= (A[r2+1] & a2_mask[a_mask_idx]) >> (a2_shift[a_mask_idx] + 3);

    a_val |= ((A[r3] & a1_mask[a_mask_idx]) << a1_shift[a_mask_idx]) >> 6;
    a_val |= (A[r3+1] & a2_mask[a_mask_idx]) >> (a2_shift[a_mask_idx] + 6);
    res += popcnt8(~(a_val^W[f_idx]));

    /* Second byte */
    a_val = 0;
    a_val |= (A[r3] & a1_mask[a_mask_idx]) << (a1_shift[a_mask_idx] + 2);
    a_val |= (A[r3+1] & a2_mask[a_mask_idx]) << (a2_b2_shift[a_mask_idx]);
    res += popcnt8(~(a_val^W[f_idx+1]));
  }

  /* Needed for xnor-popcnt */
  res = res*2 - (9*num_chan);

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

int nthbitset_arr(uint8_t *arr, int n)
{
  uint8_t x = arr[n/8];
  return x & (1 << (7-(n%8))) ? 1 : 0;
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

void printbits(uint8_t v, int n) {
  int i;
  for (i = 0; i < n-1; i++) {
    putchar('0' + ((v & 0x80)>>7));
    putchar(',');
    v <<= 1;
  }
  putchar('0' + ((v & 0x80)>>7));
}

void print_binary_mat(uint8_t *a, int M, int N, int row_major)
{
  int i, j, newline, p_len, idx;
  newline = row_major ? N : M;
  idx = 0;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; j+=8) {
      p_len = newline < j + 8 ? newline - j : 8;
      printbits(a[idx], p_len);
      if(p_len == 8 && j + 8 != newline) printf(",");
      idx++;
    }
    printf("\n");
  }
}

void print_float_mat(float *a, int M, int N)
{
  int i;

  for (i=0; i < M*N; i++)
  {
      if (i > 0 && i % M == 0) printf("\n");
      printf("%f", a[i]);
      if ((i+1) % M != 0) printf(",");
  }
  printf("\n");
}

#endif /*UTIL_H*/
