#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "tf2c.h"

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
  uint size = 4;
  assert(type == INT || type == FLOAT);
  tensor->buf = malloc(tensor->shape.size * size);
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

#define INSTANTIATE2(ret, name, args)                           \
  INSTANTIATE(ret, name, args);                                 \
  template <>                                                   \
  Tensor* name<void>(const Tensor* a, const Tensor* b) {        \
    assert(a->type == b->type);                                 \
    if (a->type == INT)                                         \
      return name<int>(a, b);                                   \
    else if (a->type == FLOAT)                                  \
      return name<float>(a, b);                                 \
    else                                                        \
      error("Unknown type: %d", a->type);                       \
  }


template <class T>
void tf2c_fill(Tensor* tensor, T v) {
  for (uint i = 0; i < tensor->shape.size; i++)
    tensor->vec<T>(i) = v;
}

template void tf2c_fill<int>(Tensor*, int);
template void tf2c_fill<float>(Tensor*, float);

template <class T>
void tf2c_assign(Tensor* tensor, const T* v) {
  for (uint i = 0; i < tensor->shape.size; i++)
    tensor->vec<T>(i) = v[i];
}

template void tf2c_assign<int>(Tensor*, const int*);
template void tf2c_assign<float>(Tensor*, const float*);

template <class T>
Tensor* tf2c_add(const Tensor* a, const Tensor* b) {
  check_tensor_type_eq(*a, *b);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (uint i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = a->vec<T>(i) + b->vec<T>(i);
  }
  return r;
}

INSTANTIATE2(Tensor*, tf2c_add, (const Tensor* a, const Tensor* b));

template <class T>
Tensor* tf2c_matmul(const Tensor* a, const Tensor* b) {
  if (a->shape.num_dims == 1) {
    assert(b->shape.num_dims == 2);
    assert(a->shape.dims[0] == b->shape.dims[0]);
    int in = a->shape.dims[0];
    int kn = b->shape.dims[1];
    Tensor* tensor = tf2c_tensor(a->type, tf2c_shape1(kn));
    for (int k = 0; k < kn; k++) {
      float s = 0;
      for (int i = 0; i < in; i++) {
        s += a->vec<T>(i) * b->mat<T>(i, k);
      }
      tensor->vec<T>(k) = s;
    }
    return tensor;
  } else {
    assert(a->shape.num_dims == 2);
    assert(b->shape.num_dims == 2);
    assert(a->shape.dims[a->shape.num_dims - 1] == b->shape.dims[0]);
    int in = a->shape.dims[0];
    int jn = a->shape.dims[1];
    int kn = b->shape.dims[1];
    Tensor* tensor = tf2c_tensor(a->type, tf2c_shape2(in, kn));
    tf2c_fill<float>(tensor, 0.0);
    for (int i = 0; i < in; i++) {
      for (int j = 0; j < jn; j++) {
        for (int k = 0; k < kn; k++) {
          tensor->mat<T>(i, k) += a->mat<T>(i, j) * b->mat<T>(j, k);
        }
      }
    }
    return tensor;
  }
}

INSTANTIATE2(Tensor*, tf2c_matmul, (const Tensor* a, const Tensor* b));
