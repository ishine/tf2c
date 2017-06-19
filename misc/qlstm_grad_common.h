#include "lib/tf2c.h"
#include "lib/mainutil.h"

#include <assert.h>
#include <math.h>

static inline float sigmoid(float v) {
  return 1.0 / (1.0 + exp(-v));
}

void qlstm_grad(Tensor** toi,
                Tensor** toj,
                Tensor** tof,
                Tensor** too,
                Tensor** toc) {
  Tensor* ti = i_read();
  Tensor* tj = j_read();
  Tensor* tf = f_read();
  Tensor* to = o_read();
  Tensor* tc = c_read();

  *toi = tf2c_tensor(ti->type, ti->shape);
  *toj = tf2c_tensor(ti->type, ti->shape);
  *tof = tf2c_tensor(ti->type, ti->shape);
  *too = tf2c_tensor(ti->type, ti->shape);
  *toc = tf2c_tensor(ti->type, ti->shape);
  for (uint i = 0; i < ti->shape.size; i++) {
    float vi = ti->vec<float>(i);
    float vj = tj->vec<float>(i);
    float vf = tf->vec<float>(i);
    float vo = to->vec<float>(i);
    float vc = tc->vec<float>(i);

    float svi = sigmoid(vi);
    float tvj = tanh(vj);
    float svf = sigmoid(vf);
    float svo = sigmoid(vo);
    float vnc = vc * svf + svi * tvj;
    float vncc = vnc;
    if (vncc < -0.5) vncc = -0.5;
    else if (vncc > 0.5) vncc = 0.5;

    float ovi = 0.0;
    float ovj = 0.0;
    float ovf = 0.0;
    float ovo = 0.0;
    float ovc = 0.0;
    if (vnc >= -0.5 && vnc <= 0.5) {
      ovi = svo * tvj * (1.0f - svi) * svi;
      ovj = svo * svi * (1.0f - tvj * tvj);
      ovf = svo * vc * (1.0f - svf) * svf;
      ovc = svo * svf;
    }
    ovo = vncc * (1.0f - svo) * svo;
    (*toi)->vec<float>(i) = ovi;
    (*toj)->vec<float>(i) = ovj;
    (*tof)->vec<float>(i) = ovf;
    (*too)->vec<float>(i) = ovo;
    (*toc)->vec<float>(i) = ovc;
  }
}
