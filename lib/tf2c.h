#include <stdio.h>

typedef unsigned int uint;

enum Type {
  INT, FLOAT
};

uint tf2c_type_size(Type type);

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
  T& vec(uint x) {
    return static_cast<T*>(buf)[x];
  }

  template<class T>
  const T& vec(uint x) const {
    return static_cast<T*>(buf)[x];
  }

  template<class T>
  T& mat(uint y, uint x) {
    return static_cast<T*>(buf)[y * shape.dims[1] + x];
  }

  template<class T>
  const T& mat(uint y, uint x) const {
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
Tensor* tf2c_fill(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_reshape(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_sum(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_broadcastgradientargs(const Tensor* a, const Tensor* b);

void tf2c_load(Tensor* tensor, const char* fname);

template <class T>
void tf2c_assign(Tensor* tensor, const T* v);

template <class T>
Tensor* tf2c_identity(const Tensor* a) { return (Tensor*)a; }

template <class T>
Tensor* tf2c_tanh(const Tensor* a);

template <class T>
Tensor* tf2c_sigmoid(const Tensor* a);

template <class T>
Tensor* tf2c_add(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_mul(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_minimum(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_maximum(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_matmul(const Tensor* a, const Tensor* b,
                    int transpose_a, int transpose_b);

void init();
