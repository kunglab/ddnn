#include <stdio.h>
#include "simple.h"
#include "inter.h"

int main () {
  int i;
  for (i = 0; i < 784; ++i) input[i] = x_in[i];

  l_conv_bn_bst0(input, temp1);

  for (i = 0; i < 10; ++i) {
    printf("%d %d\n", temp1[i], inter1[i]);
  }

  compute();
  printf("%d\n", output[0]);

  return 0;
}
