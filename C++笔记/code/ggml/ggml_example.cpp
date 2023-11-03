#include "ggml.h"
#include <iostream>
#include <stdio.h>
#include <thread>

struct ggml_init_params params = {
    .mem_size = 1600 * 1024 * 1024,
    .mem_buffer = NULL,
};

void forward1() {
  struct ggml_context *ctx = ggml_init(params);
  struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_param(ctx, x); // x is an input variable
  struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  // f(x) = a*x^2 + b
  struct ggml_tensor *x2 = ggml_mul(ctx, x, x);
  struct ggml_tensor *f = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
  struct ggml_cgraph gf = ggml_build_forward(f);
  // set the input variable and parameter values
  ggml_set_f32(x, 2.0f);
  ggml_set_f32(a, 3.0f);
  ggml_set_f32(b, 4.0f);
  // 实际计算
  int n_threads = 1;
  ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
  printf("f = %f\n", ggml_get_f32_1d(f, 0));
}

void forward2() {
  struct ggml_context *ctx = ggml_init(params);
  struct ggml_cgraph gf = {};
  struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 5);
  struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 6);
  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
  // f(x) = ax + b
  struct ggml_tensor *t = ggml_mul_mat(ctx, a, x);
  struct ggml_tensor *f = ggml_add(ctx, t, b);
  ggml_build_forward_expand(&gf, f);
  // 实际计算
  int n_threads = 1;
  ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
}

void forward3() {
  struct ggml_context *ctx = ggml_init(params);
  struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_param(ctx, x); // x is an input variable
  struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  // f(x) = a*x^2 + b
  struct ggml_tensor *x2 = ggml_mul(ctx, x, x);
  struct ggml_tensor *f = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
  struct ggml_cgraph gf = ggml_build_forward(f);
  // set the input variable and parameter values
  ggml_set_f32(x, 2.0f);
  ggml_set_f32(a, 3.0f);
  ggml_set_f32(b, 4.0f);
  // 实际计算
  int n_threads = 1;
  struct ggml_cplan plan = ggml_graph_plan(&gf, n_threads);
  std::cout << "plan.work_size:" << plan.work_size<<"\n";
  if (plan.work_size > 0) {
    plan.work_data = new uint8_t[plan.work_size];
  }
  ggml_graph_compute(&gf, &plan);
  printf("f = %f\n", ggml_get_f32_1d(f, 0));
}

int main() {
  forward1();
  forward2();
  forward3();
  return 0;
}
