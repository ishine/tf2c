#include <stdio.h>
#include <string.h>
#include <time.h>

#include "lib/mainutil.h"
#include "lib/tf2c.h"

Tensor* result();

int main(int argc, const char* argv[]) {
  init();
  run(argc, argv, result);
}
