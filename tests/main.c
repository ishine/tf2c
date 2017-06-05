#include "lib/tf2c.h"

#include <time.h>

Tensor* result();

int main() {
  init();
  clock_t start = clock();
  Tensor* tensor = result();
  double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
  dump_tensor(*tensor);
  printf("%f\n", elapsed);
}

