#include <stdio.h>

typedef unsigned int uint;

enum Type {
  INT, FLOAT
};

#define MAX_DIMS 6

struct Shape {
  uint dims[MAX_DIMS];
  uint num_dims;
  uint size;
};

struct Tensor {
  Shape shape;
  Type type;
  void* buf;
  void* alloc;

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

  template<class T>
  const T& mat(int y, int x) const {
    return static_cast<T*>(buf)[y * shape.dims[1] + x];
  }
};

Shape tf2c_shape(const int* dims);
Shape tf2c_shape0();
Shape tf2c_shape1(int d0);
Shape tf2c_shape2(int d0, int d1);

Tensor* tf2c_tensor(Type type, Shape shape);

void dump_shape(const Shape& shape);

void dump_tensor(const Tensor& tensor);

template <class T>
void tf2c_fill(Tensor* tensor, T v);

template <class T>
void tf2c_assign(Tensor* tensor, const T* v);

template <class T>
Tensor* tf2c_identity(const Tensor* a) { return (Tensor*)a; }

template <class T>
Tensor* tf2c_add(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_matmul(const Tensor* a, const Tensor* b);

void init();
