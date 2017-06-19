#include "lib/tf2c.h"
#include "lib/mainutil.h"
#include "out/qlstm_grad_i.c"

#include <assert.h>
#include <math.h>

static inline float sigmoid(float v) {
  return 1.0 / (1.0 + exp(-v));
}

Tensor* qlstm_grad_i() {
  Tensor* ti = i_read();
  Tensor* tj = j_read();
  Tensor* tf = f_read();
  Tensor* to = o_read();
  Tensor* tc = c_read();

  Tensor* toi = tf2c_tensor(ti->type, ti->shape);
  for (uint i = 0; i < ti->shape.size; i++) {
    float vi = ti->vec<float>(i);
    float vj = tj->vec<float>(i);
    float vf = tf->vec<float>(i);
    float vo = to->vec<float>(i);
    float vc = tc->vec<float>(i);

    float tvj = tanh(vj);
    float svi = sigmoid(vi);
    float svo = sigmoid(vo);
    float vnc = vc * sigmoid(vf) + svi * tanh(vj);

    float evi = 0.0;
    if (vnc >= -0.5 && vnc <= 0.5) {
      evi = (1.0f - svi) * svi;
      evi = tvj * evi;
      evi = svo * evi;
    }
    toi->vec<float>(i) = evi;
  }

  return toi;
}

int main(int argc, const char* argv[]) {
  init();
  run(argc, argv, qlstm_grad_i);
}
