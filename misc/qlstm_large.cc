#include "lib/tf2c.h"
#include "out/qlstm_large.c"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static inline float sigmoid(float v) {
  return 1.0 / (1.0 + exp(-v));
}

Tensor* qlstm() {
  Tensor* ti = i_read();
  Tensor* tj = j_read();
  Tensor* tf = f_read();
  Tensor* to = o_read();
  Tensor* tc = c_read();

  Tensor* tnc = tf2c_tensor(ti->type, ti->shape);
  Tensor* tnh = tf2c_tensor(ti->type, ti->shape);

  for (uint i = 0; i < ti->shape.size; i++) {
    float vi = ti->vec<float>(i);
    float vj = tj->vec<float>(i);
    float vf = tf->vec<float>(i);
    float vo = to->vec<float>(i);
    float vc = tc->vec<float>(i);
    float vnc = vc * sigmoid(vf) + sigmoid(vi) * tanh(vj);
    if (vnc < -0.5) vnc = -0.5;
    else if (vnc > 0.5) vnc = 0.5;
    tnc->vec<float>(i) = vnc;

    float vnh = vnc * sigmoid(vo);
    tnh->vec<float>(i) = vnh;
  }

  return tnh;
}

#define result qlstm

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
