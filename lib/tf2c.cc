#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "tf2c.h"

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
  Shape shape;
  shape.size = 1;
  shape.num_dims = 0;
  for (uint i = 0; dims[i] >= 0; i++) {
    if (i >= MAX_DIMS)
      error("more than %d dims", MAX_DIMS);
    shape.dims[i] = dims[i];
    shape.num_dims++;
    shape.size *= dims[i];
  }
  return shape;
}

Tensor* tf2c_tensor(Type type, Shape shape) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  tensor->type = type;
  tensor->shape = shape;
  tensor->buf = malloc(tensor->shape.size);
  return tensor;
}

void dump_shape(const Shape& shape) {
  printf("[");
  for (uint i = 0; i < shape.num_dims; i++) {
    if (i)
      printf(", ");
    printf("%d", shape.dims[i]);
  }
  printf("]\n");
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

INSTANTIATE(Tensor*, tf2c_add, (const Tensor* a, const Tensor* b));
