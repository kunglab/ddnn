#include <stdio.h>
#include "minunit.h"
#include "simple.h"
#include "inter.h"

int tests_run = 0;
char output_buf[1000];

static char* test_l1() {
  char output_msg[] = "Output Mismatch: Layer 1.\nComputed[%d]=%d, Actual[%d]=%d\n";
  int i;
  for (i = 0; i < 784; ++i) input[i] = x_in[i];

  l_conv_bn_bst0(input, temp1);

  for (i = 0; i < 224; ++i) {
    sprintf(output_buf, output_msg, i, temp1[i], i, inter1[i]);
    mu_assert(output_buf, temp1[i] == inter1[i]);
    /* printf("%d, %d\n", temp1[i], inter1[i]); */
  }

  return 0;
}

static char* all_tests() {
  mu_run_test(test_l1);
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


  /* compute(); */
  /* printf("%d\n", output[0]); */
  return 0;
}
