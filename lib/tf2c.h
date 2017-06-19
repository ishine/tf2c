#ifndef TF2C_H_
#define TF2C_H_

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
Tensor* tf2c_Fill(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_Reshape(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_Sum(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_BroadcastGradientArgs(const Tensor* a, const Tensor* b);

void tf2c_load(Tensor* tensor, const char* fname);

template <class T>
void tf2c_assign(Tensor* tensor, const T* v);

template <class T>
Tensor* tf2c_Identity(const Tensor* a) { return (Tensor*)a; }

template <class T>
Tensor* tf2c_Tanh(const Tensor* a);

template <class T>
Tensor* tf2c_Sigmoid(const Tensor* a);

template <class T>
Tensor* tf2c_Add(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_Mul(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_Minimum(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_Maximum(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_LessEqual(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_GreaterEqual(const Tensor* a, const Tensor* b);

template <class T>
Tensor* tf2c_MatMul(const Tensor* a, const Tensor* b,
                    int transpose_a, int transpose_b);

template <class T>
Tensor* tf2c_Select(const Tensor* a, const Tensor* b, const Tensor* c);

template <class T>
Tensor* tf2c_SigmoidGrad(const Tensor* a, const Tensor* b);

void init();

#endif  // TF2C_H_
