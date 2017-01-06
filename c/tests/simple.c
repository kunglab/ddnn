#include <stdio.h>
#include "minunit.h"
#include "simple.h"
#include "simple_compare.h"

int tests_run = 0;
char output_buf[1000];
int errors = 0;

static char* test_l1() {
  char output_msg[] = "Output Mismatch: Layer 1.\nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;
  for (i = 0; i < 784; ++i) input[i] = x_in[i];

  l_conv_bn_bst0(input, temp1);

  for (i = 0; i < 224; ++i) {
    sprintf(output_buf, output_msg, i, temp1[i], i, inter1[i]);
    mu_assert(output_buf, temp1[i] == inter1[i]);
    printf("%d, %d\n", temp1[i], inter1[i]);
  }

  return 0;
}

static char* test_l2() {
  char output_msg[] = "Output Mismatch: Layer 2.\nComputed: %d, Actual: %d\n";

  linear_BN_softmax_layer(inter1, l_b_linear_bn_softmax1_bl_W, output,
                          l_b_linear_bn_softmax1_bl_b, l_b_linear_bn_softmax1_bn_gamma,
                          l_b_linear_bn_softmax1_bn_beta, l_b_linear_bn_softmax1_bn_mean,
                          l_b_linear_bn_softmax1_bn_std, 1, 1568, 10);

  sprintf(output_buf, output_msg, output[0], 3);
  mu_assert(output_buf,  output[0] == 3);

  return 0;
}


static char* all_tests() {
  mu_run_test(test_l1);
  mu_run_test(test_l2);
  return 0;
}

int main () {
  char *result = all_tests();
  if (result != 0) {
    printf("%s\n", result);
  }
  else {
    printf("ALL TESTS PASSED\n");
  }
  printf("Tests run: %d\n", tests_run);

  return 0;
}
