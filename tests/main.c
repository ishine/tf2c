#include "lib/tf2c.h"

#include <time.h>

Tensor* result();

int main() {
  clock_t start = clock();
  init();
  Tensor* tensor = result();
  double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
  dump_tensor(*tensor);
  printf("%f\n", elapsed);
}

