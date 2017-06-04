#include "lib/tf2c.h"

Tensor* result();

int main() {
  init();
  Tensor* tensor = result();
  dump_tensor(*tensor);
}

