#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "tf2c.h"

using namespace std;

#ifdef __GNUC__
#define NORETURN __attribute__((noreturn))
#else
#define NORETURN
#endif

NORETURN
static void error(const char* fmt, ...) {
  va_list ap;
  char buf[4096];
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s\n", buf);
  exit(1);
}

uint tf2c_type_size(Type type) {
  assert(type == INT || type == FLOAT);
  return 4;
}

static void check_shape_eq(const Shape& a, const Shape& b) {
  assert(a.size == b.size);
  assert(a.num_dims == b.num_dims);
  for (uint i = 0; i < a.num_dims; i++)
    assert(a.dims[i] == b.dims[i]);
}

static void check_tensor_type_eq(const Tensor& a, const Tensor& b) {
  assert(a.type == b.type);
  check_shape_eq(a.shape, b.shape);
}

Shape tf2c_shape(const int* dims) {
  Shape shape = tf2c_shape0();
  for (uint i = 0; dims[i] >= 0; i++) {
    if (i >= MAX_DIMS)
      error("more than %d dims", MAX_DIMS);
    shape.dims[i] = dims[i];
    shape.num_dims++;
    shape.size *= dims[i];
  }
  return shape;
}

Shape tf2c_shape_from_tensor(const Tensor* a) {
  assert(a->type == INT);
  assert(a->shape.num_dims == 1);
  assert(a->shape.size <= MAX_DIMS);
  int dims[MAX_DIMS + 1];
  for (uint i = 0; i < a->shape.size; i++)
    dims[i] = a->vec<uint>(i);
  dims[a->shape.size] = -1;
  return tf2c_shape(dims);
}

Shape tf2c_shape0() {
  Shape shape;
  shape.size = 1;
  shape.num_dims = 0;
  return shape;
}

Shape tf2c_shape1(int d0) {
  Shape shape;
  shape.size = d0;
  shape.num_dims = 1;
  shape.dims[0] = d0;
  return shape;
}

Shape tf2c_shape2(int d0, int d1) {
  Shape shape;
  shape.size = d0 * d1;
  shape.num_dims = 2;
  shape.dims[0] = d0;
  shape.dims[1] = d1;
  return shape;
}

Tensor* tf2c_tensor(Type type, Shape shape) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  tensor->type = type;
  tensor->shape = shape;
  uint size = tensor->shape.size * tf2c_type_size(type);
  tensor->alloc = malloc(size + 63);
  tensor->buf = (void*)(((uintptr_t)tensor->alloc + 63) & ~63);
  return tensor;
}

void dump_shape(const Shape& shape) {
  printf("(");
  for (uint i = 0; i < shape.num_dims; i++) {
    if (i)
      printf(", ");
    printf("%d", shape.dims[i]);
  }
  printf(")\n");
}

void dump_tensor(const Tensor& tensor) {
  dump_shape(tensor.shape);
  printf("[");
  for (uint i = 0; i < tensor.shape.size; i++) {
    if (i)
      printf(", ");
    switch (tensor.type) {
    case INT:
      printf("%d", tensor.vec<int>(i));
      break;

    case FLOAT:
      printf("%f", tensor.vec<float>(i));
      break;
    }
  }
  printf("]\n");
}

#define INSTANTIATE(ret, name, args)            \
  template ret name<int> args;                  \
  template ret name<float> args;

#define INSTANTIATE1(name)                                      \
  INSTANTIATE(Tensor*, name, (const Tensor* a));                \
  template <>                                                   \
  Tensor* name<void>(const Tensor* a) {                         \
    if (a->type == INT)                                         \
      return name<int>(a);                                      \
    else if (a->type == FLOAT)                                  \
      return name<float>(a);                                    \
    else                                                        \
      error("Unknown type: %d", a->type);                       \
  }

#define INSTANTIATE2(name, d)                                       \
  INSTANTIATE(Tensor*, name, (const Tensor* a, const Tensor* b));   \
  template <>                                                       \
  Tensor* name<void>(const Tensor* a, const Tensor* b) {            \
    if (d->type == INT)                                             \
      return name<int>(a, b);                                       \
    else if (d->type == FLOAT)                                      \
      return name<float>(a, b);                                     \
    else                                                            \
      error("Unknown type: %d", d->type);                           \
  }

template <class T>
void tf2c_fill(Tensor* tensor, T v) {
  for (uint i = 0; i < tensor->shape.size; i++)
    tensor->vec<T>(i) = v;
}

void tf2c_load(Tensor* tensor, const char* fname) {
  FILE* fp = fopen(fname, "rb");
  fread(tensor->buf, tf2c_type_size(tensor->type), tensor->shape.size, fp);
}

template void tf2c_fill<int>(Tensor*, int);
template void tf2c_fill<float>(Tensor*, float);

template <class T>
Tensor* tf2c_Fill(const Tensor* a, const Tensor* b) {
  assert(b->shape.size == 1);
  Tensor* r = tf2c_tensor(a->type, tf2c_shape_from_tensor(a));
  tf2c_fill(r, b->vec<T>(0));
  return r;
}

INSTANTIATE2(tf2c_Fill, b);

template <class T>
Tensor* tf2c_Reshape(const Tensor* a, const Tensor* b) {
  Tensor* r = tf2c_Identity<void>(a);
  r->shape = tf2c_shape_from_tensor(b);
  return r;
}

INSTANTIATE2(tf2c_Reshape, a);

template <class T>
Tensor* tf2c_Sum(const Tensor* a, const Tensor*) {
  // TODO
  return (Tensor*)a;
}

INSTANTIATE2(tf2c_Sum, a);

template <class T>
Tensor* tf2c_BroadcastGradientArgs(const Tensor*, const Tensor*) {
  // TODO
  return nullptr;
}

INSTANTIATE2(tf2c_BroadcastGradientArgs, a);

template <class T>
void tf2c_assign(Tensor* tensor, const T* v) {
  for (uint i = 0; i < tensor->shape.size; i++)
    tensor->vec<T>(i) = v[i];
}

template void tf2c_assign<int>(Tensor*, const int*);
template void tf2c_assign<float>(Tensor*, const float*);

template <class T>
Tensor* tf2c_Tanh(const Tensor* a) {
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = tanh(a->vec<T>(i));
  }
  return r;
}

INSTANTIATE1(tf2c_Tanh);

template <class T>
Tensor* tf2c_Sigmoid(const Tensor* a) {
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = 1.0 / (1.0 + exp(-a->vec<T>(i)));
  }
  return r;
}

INSTANTIATE1(tf2c_Sigmoid);

#ifdef __AVX2__

Tensor* tf2c_Add_tile_avx(const Tensor* a, const Tensor* b) {
#if 0

  if (a->type != FLOAT && a->shape.dims[2] % 8 == 0)
    return nullptr;
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.dims[0]; i++) {
    for (uint j = 0; j < a->shape.dims[1]; j++) {
      for (uint k = 0; k < a->shape.dims[2]; k += 8) {
        uint ia = (i * a->shape.dims[1] + j) * a->shape.dims[2] + k;
        uint ib = i * a->shape.dims[2] + k;
        __m256 av = _mm256_loadu_ps(&a->vec<float>(ia));
        __m256 bv = _mm256_loadu_ps(&b->vec<float>(ib));
        _mm256_storeu_ps(&r->vec<float>(ia), _mm256_add_ps(av, bv));
      }
    }
  }
  return r;

#else

  static const uint KS = 4;
  if (a->type != FLOAT && a->shape.dims[2] % (8 * KS) == 0)
    return nullptr;
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.dims[0]; i++) {
    __m256 bv[KS] = {};
    for (uint k1 = 0; k1 < a->shape.dims[2]; k1 += 8 * KS) {
      for (uint k2 = 0; k2 < KS; k2++) {
        uint ib = i * a->shape.dims[2] + k1 + k2 * 8;
        bv[k2] = _mm256_loadu_ps(&b->vec<float>(ib));
      }

      for (uint j = 0; j < a->shape.dims[1]; j++) {
        for (uint k2 = 0; k2 < KS; k2++) {
          uint k = k1 + k2 * 8;
          uint ia = (i * a->shape.dims[1] + j) * a->shape.dims[2] + k;
          __m256 av = _mm256_loadu_ps(&a->vec<float>(ia));
          _mm256_storeu_ps(&r->vec<float>(ia), _mm256_add_ps(av, bv[k2]));
        }
      }
    }
  }
  return r;

#endif
}

#endif

template <class T>
Tensor* tf2c_Add(const Tensor* a, const Tensor* b) {
  assert(a->type == b->type);
  assert(a->shape.num_dims == b->shape.num_dims);
  if (a->shape.size != b->shape.size) {
    assert(a->shape.dims[0] == b->shape.dims[0]);
    assert(a->shape.dims[1] > b->shape.dims[1]);
    assert(a->shape.dims[2] == b->shape.dims[2]);
    Tensor* r = nullptr;
#ifdef __AVX2__
    r = tf2c_Add_tile_avx(a, b);
    if (r)
      return r;
#endif
    r = tf2c_tensor(a->type, a->shape);
    for (uint i = 0; i < a->shape.dims[0]; i++) {
      for (uint j = 0; j < a->shape.dims[1]; j++) {
        for (uint k = 0; k < a->shape.dims[2]; k++) {
          uint ia = (i * a->shape.dims[1] + j) * a->shape.dims[2] + k;
          uint ib = i * a->shape.dims[2] + k;
          r->vec<T>(ia) = a->vec<T>(ia) + b->vec<T>(ib);
        }
      }
    }
    return r;
  } else {
    check_tensor_type_eq(*a, *b);
    Tensor* r = tf2c_tensor(a->type, a->shape);
    for (uint i = 0; i < a->shape.size; i++) {
      r->vec<T>(i) = a->vec<T>(i) + b->vec<T>(i);
    }
    return r;
  }
}

INSTANTIATE2(tf2c_Add, a);

template <class T>
Tensor* tf2c_Mul(const Tensor* a, const Tensor* b) {
  check_tensor_type_eq(*a, *b);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = a->vec<T>(i) * b->vec<T>(i);
  }
  return r;
}

INSTANTIATE2(tf2c_Mul, a);

template <class T>
Tensor* tf2c_Minimum(const Tensor* a, const Tensor* b) {
  assert(b->shape.size == 1);
  T v = b->vec<T>(0);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = min(a->vec<T>(i), v);
  }
  return r;
}

INSTANTIATE2(tf2c_Minimum, a);

template <class T>
Tensor* tf2c_Maximum(const Tensor* a, const Tensor* b) {
  assert(b->shape.size == 1);
  T v = b->vec<T>(0);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = max(a->vec<T>(i), v);
  }
  return r;
}

INSTANTIATE2(tf2c_Maximum, a);

template <class T>
Tensor* tf2c_LessEqual(const Tensor* a, const Tensor* b) {
  assert(b->shape.size == 1);
  T v = b->vec<T>(0);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = static_cast<T>(a->vec<T>(i) <= v);
  }
  return r;
}

INSTANTIATE2(tf2c_LessEqual, a);

template <class T>
Tensor* tf2c_GreaterEqual(const Tensor* a, const Tensor* b) {
  assert(b->shape.size == 1);
  T v = b->vec<T>(0);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = static_cast<T>(a->vec<T>(i) >= v);
  }
  return r;
}

INSTANTIATE2(tf2c_GreaterEqual, a);

#ifdef __AVX2__

static bool tf2c_matmul_avx2(const Tensor* a, const Tensor* b, Tensor* r) {
  uint in = a->shape.dims[0];
  uint jn = a->shape.dims[1];
  uint kn = b->shape.dims[1];
  static const uint IS = 4;
  static const uint KS = 2;
  if (in % IS != 0 || kn % (KS * 8) != 0)
    return false;
  // broadcast: I*J*K = 128M
  // fma: I*J*K/8 = 128M
  // load: I*J*K = 128M
  for (uint i = 0; i < in; i += IS) {
    for (uint k = 0; k < kn; k += KS * 8) {
      __m256 rv[IS][KS] __attribute__((aligned(32))) = { 0 };
      for (uint j = 0; j < jn; j++) {
        for (uint i2 = 0; i2 < IS; i2++) {
          for (uint k2 = 0; k2 < KS; k2++) {
            rv[i2][k2] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&a->mat<float>(i + i2, j)),
                _mm256_loadu_ps(&b->mat<float>(j, k + k2 * 8)),
                rv[i2][k2]);
          }
        }
      }

      for (uint i2 = 0; i2 < IS; i2++) {
        for (uint k2 = 0; k2 < KS; k2++) {
          _mm256_storeu_ps(
              &r->mat<float>(i + i2, k + k2 * 8),
              _mm256_add_ps(
                  _mm256_loadu_ps(&r->mat<float>(i + i2, k + k2 * 8)),
                  rv[i2][k2]));
        }
      }
    }
  }
  return true;
}

static bool tf2c_vecmatmul_avx2(const Tensor* a, const Tensor* b, Tensor* r) {
  uint in = a->shape.dims[0];
  if (in != 1)
    return false;
  uint jn = a->shape.dims[1];
  uint kn = b->shape.dims[1];
  static const uint KS = 4;
  if (kn % (KS * 8) != 0)
    return false;
#if 0
  // Not good for cache.
  for (uint k = 0; k < kn; k += 8) {
    __m256 rv = {0};
    for (uint j = 0; j < jn; j++) {
      __m256 av = _mm256_broadcast_ss(&a->mat<float>(0, j));
      rv = _mm256_fmadd_ps(av,
                           _mm256_loadu_ps(&b->mat<float>(j, k)),
                           rv);
    }
    _mm256_storeu_ps(&r->mat<float>(0, k), rv);
  }

#elif 1
  // broadcast: J = 3k
  // fma: J*K/8 = 1.5M
  // store: J*K/8 = 1.5M
  // load: J*K/4 = 3M
  for (uint j = 0; j < jn; j++) {
    __m256 av = _mm256_broadcast_ss(&a->mat<float>(0, j));
    for (uint k = 0; k < kn; k += 8) {
      _mm256_storeu_ps(
          &r->mat<float>(0, k),
          _mm256_fmadd_ps(
              av,
              _mm256_loadu_ps(&b->mat<float>(j, k)),
              _mm256_loadu_ps(&r->mat<float>(0, k))));
    }
  }

#else
  // broadcast: J = 3k
  // fma: J*K/8 = 1.5M
  // store: J*KS = 12k
  // load: J*K/8 + J*KS = 1.5M
  for (uint k = 0; k < kn; k += KS * 8) {
    __m256 rv[KS] __attribute__((aligned(32))) = { 0 };
    for (uint j = 0; j < jn; j++) {
      __m256 av = _mm256_broadcast_ss(&a->mat<float>(0, j));
      for (uint k2 = 0; k2 < KS; k2++) {
        rv[k2] = _mm256_fmadd_ps(
            av,
            _mm256_loadu_ps(&b->mat<float>(j, k + k2 * 8)),
            rv[k2]);
      }
    }

    for (uint k2 = 0; k2 < KS; k2++) {
      _mm256_storeu_ps(
          &r->mat<float>(0, k + k2 * 8),
          _mm256_add_ps(
              _mm256_loadu_ps(&r->mat<float>(0, k + k2 * 8)),
              rv[k2]));
    }
  }
#endif
  return true;
}

Tensor* tf2c_vecmatmul_trans_avx2(const Tensor* a, const Tensor* b,
                                  int transpose_a, int transpose_b) {
  if (a->type != FLOAT || transpose_a || !transpose_b)
    return nullptr;
  uint in = a->shape.dims[0];
  if (in != 1)
    return nullptr;
  assert(a->shape.dims[1] == b->shape.dims[1]);
  uint jn = b->shape.dims[0];
  uint kn = b->shape.dims[1];
  if (kn % 8 != 0)
    return nullptr;

  Tensor* r = tf2c_tensor(a->type, tf2c_shape2(1, jn));
  tf2c_fill<float>(r, 0.0);
  for (uint j = 0; j < jn; j++) {
    __m256 rv = { 0 };
    for (uint k = 0; k < kn; k += 8) {
      __m256 av = _mm256_loadu_ps(&a->vec<float>(k));
      __m256 bv = _mm256_loadu_ps(&b->mat<float>(j, k));
      rv = _mm256_fmadd_ps(av, bv, rv);
    }
    float t[8] __attribute__((aligned(32)));
    _mm256_storeu_ps(t, rv);
    float rvt = 0;
    for (uint i = 0; i < 8; i++)
      rvt += t[i];
    r->vec<float>(j) = rvt;
  }
  return r;
}

#endif

template <class T>
Tensor* tf2c_transpose_mat(const Tensor* a) {
  assert(a->shape.num_dims == 2);
  uint in = a->shape.dims[0];
  uint jn = a->shape.dims[1];
  Tensor* tensor = tf2c_tensor(a->type, tf2c_shape2(jn, in));
  for (uint i = 0; i < in; i++) {
    for (uint j = 0; j < jn; j++) {
      tensor->mat<T>(j, i) = a->mat<T>(i, j);
    }
  }
  return tensor;
}

INSTANTIATE1(tf2c_transpose_mat);

template <class T>
Tensor* tf2c_MatMul(const Tensor* a, const Tensor* b,
                    int transpose_a, int transpose_b) {
  if (a->shape.num_dims == 1) {
    assert(b->shape.num_dims == 2);
    assert(a->shape.dims[0] == b->shape.dims[0]);
    assert(!transpose_a);
    assert(!transpose_b);
    uint in = a->shape.dims[0];
    uint kn = b->shape.dims[1];
    Tensor* tensor = tf2c_tensor(a->type, tf2c_shape1(kn));
    for (uint k = 0; k < kn; k++) {
      float s = 0;
      for (uint i = 0; i < in; i++) {
        s += a->vec<T>(i) * b->mat<T>(i, k);
      }
      tensor->vec<T>(k) = s;
    }
    return tensor;
  } else {
    Tensor* tensor = nullptr;
#ifdef __AVX2__
    tensor = tf2c_vecmatmul_trans_avx2(a, b, transpose_a, transpose_b);
    if (tensor)
      return tensor;
#endif

    if (transpose_a)
      a = tf2c_transpose_mat<void>(a);
    if (transpose_b)
      b = tf2c_transpose_mat<void>(b);
    assert(a->shape.num_dims == 2);
    assert(b->shape.num_dims == 2);
    assert(a->shape.dims[a->shape.num_dims - 1] == b->shape.dims[0]);
    uint in = a->shape.dims[0];
    uint jn = a->shape.dims[1];
    uint kn = b->shape.dims[1];
    tensor = tf2c_tensor(a->type, tf2c_shape2(in, kn));
    tf2c_fill<float>(tensor, 0.0);

#ifdef __AVX2__
    if (a->type == FLOAT) {
      if (tf2c_vecmatmul_avx2(a, b, tensor))
        return tensor;
      if (tf2c_matmul_avx2(a, b, tensor))
        return tensor;
    }
#endif

    for (uint i = 0; i < in; i++) {
      for (uint j = 0; j < jn; j++) {
        for (uint k = 0; k < kn; k++) {
          tensor->mat<T>(i, k) += a->mat<T>(i, j) * b->mat<T>(j, k);
        }
      }
    }
    return tensor;
  }
}

template<>
Tensor* tf2c_MatMul<void>(const Tensor* a, const Tensor* b,
                          int transpose_a, int transpose_b) {
  assert(a->type == b->type);
  if (a->type == INT)
    return tf2c_MatMul<int>(a, b, transpose_a, transpose_b);
  else if (a->type == FLOAT)
    return tf2c_MatMul<float>(a, b, transpose_a, transpose_b);
  else
    error("Unknown type: %d", a->type);
}

template <>
Tensor* tf2c_Select<void>(const Tensor*, const Tensor*, const Tensor*) {
  error("TODO!");
  return nullptr;
}

template <>
Tensor* tf2c_SigmoidGrad<void>(const Tensor*, const Tensor*) {
  error("TODO!");
  return nullptr;
}

template <>
Tensor* tf2c_TanhGrad<void>(const Tensor*, const Tensor*) {
  error("TODO!");
  return nullptr;
}
