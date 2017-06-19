#include "mainutil.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "tf2c.h"

static double get_time() {
#if defined(__linux__)
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 0.001 * 0.001 * 0.001;
#else
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0)
    PERROR("gettimeofday");
  return tv.tv_sec + tv.tv_usec * 0.001 * 0.001;
#endif
}

void run(int argc, const char* argv[], Tensor* (*fp)()) {
  double start = get_time();
  if (argc > 1 && !strcmp(argv[1], "--bench")) {
    int n = 0;
    while (1) {
      fp();
      n++;
      double elapsed = get_time() - start;
      if (elapsed > 1.0) {
        printf("%f\n", elapsed / n);
        break;
      }
    }
  } else {
    Tensor* tensor = fp();
    double elapsed = get_time() - start;
    dump_tensor(*tensor);
    printf("%f\n", elapsed);
  }
}
