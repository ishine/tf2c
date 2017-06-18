#include "lib/tf2c.h"
#include "lib/thread_pool.h"
#include "out/qlstm_large.c"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef __AVX2__
#include <immintrin.h>
#include "lib/avx_mathfun.h"
#endif

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

static inline float sigmoid(float v) {
  return 1.0 / (1.0 + exp(-v));
}

#ifdef __AVX2__

static inline __m256 avx2_sigmoid(__m256 v) {
  v = _mm256_sub_ps(_mm256_setzero_ps(), v);
  v = exp256_ps(v);
  v = _mm256_add_ps(_mm256_set1_ps(1.0f), v);
  v = _mm256_div_ps(_mm256_set1_ps(1.0f), v);
  return v;
}

static inline __m256 avx2_tanh(__m256 v) {
  v = _mm256_mul_ps(_mm256_set1_ps(2.0f), v);
  v = avx2_sigmoid(v);
  v = _mm256_mul_ps(_mm256_set1_ps(2.0f), v);
  v = _mm256_sub_ps(v, _mm256_set1_ps(1.0f));
  return v;
}

#endif

Tensor* qlstm() {
  Tensor* ti = i_read();
  Tensor* tj = j_read();
  Tensor* tf = f_read();
  Tensor* to = o_read();
  Tensor* tc = c_read();

  Tensor* tnc = tf2c_tensor(ti->type, ti->shape);
  Tensor* tnh = tf2c_tensor(ti->type, ti->shape);

#ifdef __AVX2__
  assert(ti->shape.size % 8 == 0);

  ThreadPool thread_pool(NUM_THREADS);
  for (uint t = 0; t < NUM_THREADS; t++) {
    uint nt = (ti->shape.size / 8 + NUM_THREADS - 1) / NUM_THREADS;
    uint ib = nt * t * 8;
    uint ie = t == NUM_THREADS - 1 ? ti->shape.size : nt * (t + 1) * 8;
    thread_pool.Submit([ib, ie, ti, tj, tf, to, tc, tnc, tnh]() {
        __m256 clip_min = _mm256_set1_ps(-0.5);
        __m256 clip_max = _mm256_set1_ps(0.5);
        for (uint i = ib; i < ie; i += 8) {
          __m256 vi = _mm256_loadu_ps(&ti->vec<float>(i));
          __m256 vj = _mm256_loadu_ps(&tj->vec<float>(i));
          __m256 vf = _mm256_loadu_ps(&tf->vec<float>(i));
          __m256 vo = _mm256_loadu_ps(&to->vec<float>(i));
          __m256 vc = _mm256_loadu_ps(&tc->vec<float>(i));

          __m256 vnc = _mm256_add_ps(
              _mm256_mul_ps(vc, avx2_sigmoid(vf)),
              _mm256_mul_ps(avx2_sigmoid(vi), avx2_tanh(vj)));
          __m256 min_mask = _mm256_cmp_ps(vnc, clip_min, _CMP_LT_OQ);
          vnc = _mm256_blendv_ps(vnc, clip_min, min_mask);
          __m256 max_mask = _mm256_cmp_ps(vnc, clip_max, _CMP_GT_OQ);
          vnc = _mm256_blendv_ps(vnc, clip_max, max_mask);
          _mm256_storeu_ps(&tnc->vec<float>(i), vnc);

          __m256 vnh = _mm256_mul_ps(vnc, avx2_sigmoid(vo));
          _mm256_storeu_ps(&tnh->vec<float>(i), vnh);
        }
      });
  }
  thread_pool.Wait();

#else
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
#endif

  return tnh;
}

#define result qlstm

int main(int argc, const char* argv[]) {
  init();
  double start = get_time();
  if (argc > 1 && !strcmp(argv[1], "--bench")) {
    int n = 0;
    while (1) {
      result();
      n++;
      double elapsed = get_time() - start;
      if (elapsed > 1.0) {
        printf("%f\n", elapsed / n);
        break;
      }
    }
  } else {
    Tensor* tensor = result();
    double elapsed = get_time() - start;
    dump_tensor(*tensor);
    printf("%f\n", elapsed);
  }
}
