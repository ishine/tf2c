#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void error(const char* fmt, ...) {
  va_list ap;
  char buf[4096];
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s\n", buf);
  exit(1);
}

typedef unsigned int uint;

enum Type {
  INT, FLOAT
};

#define MAX_DIMS 6

struct Shape {
  uint dims[MAX_DIMS];
  uint num_dims;
  uint size;

  static void check_eq(const Shape& a, const Shape& b) {
    assert(a.size == b.size);
    assert(a.num_dims == b.num_dims);
    for (uint i = 0; i < a.num_dims; i++)
      assert(a.dims[i] == b.dims[i]);
  }
};

struct Tensor {
  Shape shape;
  Type type;
  void* buf;

  static void check_type_eq(const Tensor& a, const Tensor& b) {
    assert(a.type == b.type);
    Shape::check_eq(a.shape, b.shape);
  }

  template<class T>
  T& vec(int x) {
    return static_cast<T*>(buf)[x];
  }

  template<class T>
  const T& vec(int x) const {
    return static_cast<T*>(buf)[x];
  }

  template<class T>
  T& mat(int y, int x) {
    return static_cast<T*>(buf)[y * shape.dims[1] + x];
  }
};

Shape tf2c_shape(const int* dims) {
  Shape shape;
  shape.size = 1;
  shape.num_dims = 0;
  for (int i = 0; dims[i] >= 0; i++) {
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

template <class T>
Tensor* tf2c_add(const Tensor* a, const Tensor* b) {
  Tensor::check_type_eq(*a, *b);
  Tensor* r = tf2c_tensor(a->type, a->shape);
  for (int i = 0; i < a->shape.size; i++) {
    r->vec<T>(i) = a->vec<T>(i) + b->vec<T>(i);
  }
  return r;
}

void init();
