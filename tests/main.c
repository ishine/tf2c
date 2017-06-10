#include "lib/tf2c.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

Tensor* result();

int main(int argc, const char* argv[]) {
  init();
  clock_t start = clock();
  if (argc > 1 && !strcmp(argv[1], "--bench")) {
    int n = 0;
    while (1) {
      result();
      n++;
      double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
      if (elapsed > 1.0) {
        printf("%f\n", elapsed / n);
        break;
      }
    }
  } else {
    Tensor* tensor = result();
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    dump_tensor(*tensor);
    printf("%f\n", elapsed);
  }
}

