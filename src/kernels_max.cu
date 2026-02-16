#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// 参考 https://github.com/Jason-Young123/Learning-CUDA

/*
  1. 独立的stream + async加载/拷贝/释放
  2. 多次cudaMalloc/cudaFree合并
  3. 仅加载有效数据(对角元素)至GM; 将Qi, Ki, Vi从GM加载至SM再连续访问/运算
  4. 多利用同一warp内的shfl_down_sync进行归约
  5. 短循环#pragma unroll展开; 分支条件(branch-resolving)优化
  6. SM资源复用, 如S和P矩阵
  7. 中间关键步骤用double以保护精度(如S/P矩阵和scale_factor)
*/

static cudaStream_t stream1 = nullptr;
static cudaStream_t stream2 = nullptr;

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

template <typename T>
__device__ T warp_reduce_sum(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void trace_calc(T* d_trace, const T* d_diag, size_t n){
  __shared__ T smem[32];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  //通过两级归约(warp内, block内/warp间)完成所有元素相加
  T sum = T(0);
  for(size_t i = idx; i < n; i += stride){
    sum += d_diag[i];
  }

  //一级归约,每个warp内完成规约
  T warp_sum = warp_reduce_sum(sum);

  //准备二级规约,将每个warp内lane[0]的值拷贝到smem中
  if((tid % 32) == 0){
      smem[tid / 32] = warp_sum;
  }
  __syncthreads();//等待拷贝至smem操作完成

  //需保证一级归约后每个block内的线程不超过32(即block_dim不超过32*32 = 1024)
  if(tid < 32){
      //准备二级规约,多余线程补0
      T block_sum = (tid < (blockDim.x + 31)/32) ? smem[tid] : T(0);
      //二级归约
      block_sum = warp_reduce_sum(block_sum);
      if(tid == 0 && block_sum != T(0)){//原子操作次数 = block数量
        atomicAdd(d_trace, block_sum);
      }
  }  
  return;
}


//提取对角元素
template <typename T>
std::vector<T> extract_diag(const std::vector<T> & h_input, size_t rows, size_t cols){
  size_t n = std::min(rows, cols);
  std::vector<T> diag(n);
  for(size_t i = 0; i < n; ++i){
    diag[i] = h_input[i * cols + i];
  }
  return diag;
}



#ifdef PLATFORM_NVIDIA
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  //step-0: basic check
  //printf("=== Nvidia Device Properties ===\n");
  //cudaDeviceProp prop; 
  //int device_id = 0; 
  //cudaGetDeviceProperties(&prop, device_id);
  
  // 静态property
  //printf("Device Name: %s\n", prop.name);
  //printf("  - Max Threads per Block: %d\n", prop.maxThreadsPerBlock);//1024
  //printf("  - Max Block Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);//(1024, 1024, 64)
  //printf("  - Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);//(2147483647, 65535, 65535)
  //printf("  - Warp Size: %d\n", prop.warpSize);//32 
  //printf("  - Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);//48KB
  //printf("  - Number of Multiprocessors: %d\n", prop.multiProcessorCount);//108

  if(!std::min(rows, cols)){
    return T(0);
  }

  //step-1: 提取对角元
  std::vector<T> h_diag = extract_diag<T>(h_input, rows, cols);

  //step-2: 初始化,分配device端空间
  //cudaStream_t stream1;
  cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, -1);

  const size_t size_bytes = h_diag.size() * sizeof(T);
  const size_t total_bytes = (h_diag.size() + 1) * sizeof(T);
  T *d_all = nullptr;
  RUNTIME_CHECK(cudaMallocAsync(&d_all, total_bytes, stream1));
  T *d_diag = d_all;
  T *d_trace = d_all + h_diag.size();

  //step-3: 拷贝数据from host to device
  RUNTIME_CHECK(cudaMemcpyAsync(d_diag, h_diag.data(), size_bytes, cudaMemcpyHostToDevice, stream1));
  RUNTIME_CHECK(cudaMemsetAsync(d_trace, 0, sizeof(T), stream1));

  //step-4: device端计算
  int block_dim = 1024;
  int grid_dim = std::min((h_diag.size() + block_dim - 1)/block_dim, size_t(8));//设置上限
  trace_calc<T><<<grid_dim, block_dim, 0, stream1>>>(d_trace, d_diag, h_diag.size());//调用device端函数进行trace计算, 注意核函数返回类型只能为void
  RUNTIME_CHECK(cudaStreamSynchronize(stream1));//important

  //step-5: 拷贝数据from device to host
  T h_trace = T(0);
  RUNTIME_CHECK(cudaMemcpyAsync(&h_trace, d_trace, sizeof(T), cudaMemcpyDeviceToHost, stream1));
  RUNTIME_CHECK(cudaStreamSynchronize(stream1));//important

  //step5: free memory
  RUNTIME_CHECK(cudaFreeAsync(d_all, stream1));

  //std::cout << "h_trace is: " << h_trace << std::endl;
  return h_trace;
}
#endif




#ifdef PLATFORM_ILUVATAR//iluvatar平台，去除async资源分配
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  //step-0: basic check
  //printf("=== Nvidia Device Properties ===\n");
  //cudaDeviceProp prop; 
  //int device_id = 0; 
  //cudaGetDeviceProperties(&prop, device_id);

  // 静态property
  //printf("Device Name: %s\n", prop.name);
  //printf("  - Max Threads per Block: %d\n", prop.maxThreadsPerBlock);//1024
  //printf("  - Max Block Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);//(1024, 1024, 64)
  //printf("  - Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);//(2147483647, 65535, 65535)
  //printf("  - Warp Size: %d\n", prop.warpSize);//32 
  //printf("  - Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);//48KB
  //printf("  - Number of Multiprocessors: %d\n", prop.multiProcessorCount);//108

  if(!std::min(rows, cols)){
    //std::cerr << "Matrix Shape Invalid" << std::endl;
    return T(0);
  }

  //step-1: 提取对角元
  std::vector<T> h_diag = extract_diag<T>(h_input, rows, cols);

  //step-2: 初始化,分配device端空间
  const size_t size_bytes = h_diag.size() * sizeof(T);
  T *d_diag, *d_trace;//device端只支持裸指针
  RUNTIME_CHECK(cudaMalloc(&d_diag, size_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_trace, sizeof(T)));

  //step-3: 拷贝数据from host to device
  RUNTIME_CHECK(cudaMemcpy(d_diag, h_diag.data(), size_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_trace, 0, sizeof(T)));

  //step-4: device端计算
  int block_dim = 1024;
  int grid_dim = std::min((h_diag.size() + block_dim - 1)/block_dim, size_t(8));//设置上限
  trace_calc<T><<<grid_dim, block_dim>>>(d_trace, d_diag, h_diag.size());//调用device端函数进行trace计算
  //注意核函数返回类型只能为void

  //step-5: 拷贝数据from device to host
  T h_trace = T(0);
  RUNTIME_CHECK(cudaMemcpy(&h_trace, d_trace, sizeof(T), cudaMemcpyDeviceToHost));

  //step5: free memory
  RUNTIME_CHECK(cudaFree(d_diag));
  RUNTIME_CHECK(cudaFree(d_trace));

  //printf("the result is %f\n", float(h_trace));
  return h_trace;
}
#endif




/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

template <typename T>
__device__ T myexp(T x) {
    if constexpr(std::is_same<T, __half>::value) {
        float fx = __half2float(x);
        float result = expf(fx);
        return __float2half(result);
    }
    else if constexpr(std::is_same<T, float>::value) {
        return expf(x);  // expf返回float
    }
    else if constexpr(std::is_same<T, double>::value) {
        return exp(x);   // exp返回double
    }
    else{//other types
      return T(0);
    }
}

template <typename T>
__device__ T warp_reduce_max(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > tmp) ? val : tmp;
    }
    return val;
}


#ifdef PLATFORM_NVIDIA
//对应flash attention v2原文算法, block采用三维布局
template <typename T>
__global__ void kernel_flashAttention(int batch_size, int target_seq_len, int src_seq_len, int q_heads, int kv_heads, int head_dim, bool is_causal, const T* Q, const T* K, const T* V, T* O){
  int tid_x = threadIdx.x;//横向,blockDim.x列
  int tid_y = threadIdx.y;//纵向,blockDim.y行
  int bid_x = blockIdx.x;//x方向,总数 = #q_heads
  int bid_y = blockIdx.y;//y方向,总数 = #batch
  int bid_z = blockIdx.z;//z方向,总数 = Tr
  const int p = q_heads / kv_heads;//计算比例系数
  const int Br = blockDim.y;//Q纵向每块大小, 默认为32 (RTX 5090)
  const int Bc = blockDim.x;//K/V纵向分块大小, 默认为32
  const int Tc = (src_seq_len + Bc - 1) / Bc;//对应原始论文中K/V纵向分块数Tc,其中Bc = 32

  //预计算常量
  const double scale_factor = 1.0 / sqrt(double(head_dim));//保留精度,采用double

  //定义一系列临时变量
  /*__shared__ double SP[Br][Bc];//复用S和P
  __shared__ double m_prev[Br], m_new[Br];
  __shared__ double l_prev[Br], l_new[Br];

  __shared__ float Q_sm[Br][64];
  __shared__ float K_T_sm[64][Bc];//transpose for K
  __shared__ float V_sm[Bc][64];
  __shared__ float O_sm[Br][64];*/

  extern __shared__ char shared_mem[];
  char* ptr = shared_mem;  
  //计算中间变量,包括S, P(复用为SP), m_prev, m_new, l_prev, l_new; 为保留精度, SP采用double
  double* SP = reinterpret_cast<double*>(ptr);    // double SP[Br][Bc]
  ptr += Br * Bc * sizeof(double);
  float* m_prev = reinterpret_cast<float*>(ptr);  // float m_prev[Br]
  ptr += Br * sizeof(float);
  float* m_new = reinterpret_cast<float*>(ptr);   // float m_new[Br] 
  ptr += Br * sizeof(float);
  float* l_prev = reinterpret_cast<float*>(ptr);  // float l_prev[Br]
  ptr += Br * sizeof(float);
  float* l_new = reinterpret_cast<float*>(ptr);   // float l_new[Br] 
  ptr += Br * sizeof(float);  

  //原始数据QKV和计算结果O; 全采用float
  float* Q_sm = reinterpret_cast<float*>(ptr);    // float Q_sm[Br][head_dim] 
  ptr += Br * head_dim * sizeof(float);  
  float* K_T_sm = reinterpret_cast<float*>(ptr);  // float K_T_sm[head_dim][Bc]
  ptr += head_dim * Bc * sizeof(float);
  float* V_sm = reinterpret_cast<float*>(ptr);    // float V_sm[Br][head_dim] 
  ptr += Bc * head_dim * sizeof(float);  
  float* O_sm = reinterpret_cast<float*>(ptr);    // float O_sm[Br][head_dim]

  //定义访问宏
  /*#define   SP_AT(y, x)       SP[y][x]
  #define   Q_sm_AT(y, x)     Q_sm[y][x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y][x]
  #define   V_sm_AT(y, x)     V_sm[y][x]
  #define   O_sm_AT(y, x)     O_sm[y][x]*/

  #define   SP_AT(y, x)       SP[y * Bc + x]
  #define   Q_sm_AT(y, x)     Q_sm[y * head_dim + x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y * Bc + x]
  #define   V_sm_AT(y, x)     V_sm[y * head_dim + x]
  #define   O_sm_AT(y, x)     O_sm[y * head_dim + x]


  /****************************preparation**************************/
  int bound_tid_y = ::min(Br, target_seq_len - Br * bid_z);

  //preparation-1: load Qi from GM to SM, and reset Oi to 0
  //Q[bid_y][Br * bid_z + tid_y][bid_x][*]
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    O_sm_AT(tid_y, idx) = 0.0;
    Q_sm_AT(tid_y, idx) = 0.0;
    if(tid_y < bound_tid_y){
      Q_sm_AT(tid_y, idx) = float(Q[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx]);
    }
  }
  __syncthreads();

  //preparation-2: reset m_prev to -INFINITY and l_prev to 0
  if(tid_x == 0){
    m_prev[tid_y] = -8192.0;
    l_prev[tid_y] = 0.0;
  }
  __syncthreads();
  /****************************end-of-preparation*************************/


  /****************************main-loop**************************/
  #pragma unroll 4
  for(int j = 0; j < Tc; ++j){//对于每个K/V分块
    bool skip = (is_causal && bid_z < j);
    if(skip){//early exit, 直接跳过
    __syncthreads();
      continue;
    }

    SP_AT(tid_y, tid_x) = -8192.0;
    __syncthreads();
    int bound_tid_x = ::min(Bc, src_seq_len - Bc * j);
    bool is_compute = true;//optimization: 分支处理,加速branch-resolving
    if (is_causal) {
      if (bid_z < j) {
        is_compute = false;  // 早期退出情况
      } else if (bid_z == j) {
        is_compute = (tid_y >= tid_x);  // 对角线以上
      }
    }

    //step-1: load Ki, Vi from GM to SM, reset Oi to 0
    //K[bid_y][Bc * j + tid_y][bid_x / p][*], V[bid_y][Bc * j + tid_y][bid_x / p][*]
    #pragma unroll
    for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
      K_T_sm_AT(idx, tid_y) = 0.0;
      V_sm_AT(tid_y, idx) = 0.0;
      if(tid_y < bound_tid_x){//注意这里是bound_tid_x
        K_T_sm_AT(idx, tid_y) = float(K[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
        V_sm_AT(tid_y, idx) = float(V[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
      }
    }
    __syncthreads();

    //step-2: S = Q @ K.T, point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){//用于边缘不完整块
      float val0 = 0.0;//临时sum
      if(is_compute){
        #pragma unroll
        for(int k = 0; k < head_dim; ++k){
          val0 += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, tid_x);
        }
        SP_AT(tid_y, tid_x) = double(val0) * scale_factor;//必须用double,对精度影响最大的计算步骤
      }
    }
    __syncthreads();

    //step-3: m_new = max(m_prev, rowMax(S))
    float val1 = float(SP_AT(tid_y, tid_x));
    val1 = warp_reduce_max(val1);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val1 = SP_AT(tid_y, 0);//手动实现非并行求行最大值
      for(int h = 1; h < Bc; ++h){
        val1 = (val1 < SP_AT(tid_y, h)) ? SP_AT(tid_y, h) : val1;
      }*/
      m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
    }
    __syncthreads();

    //step-4: P = exp(S - m_new), point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){
      if(is_compute){
        SP_AT(tid_y, tid_x) = myexp<double>(SP_AT(tid_y, tid_x) - double(m_new[tid_y]));
      }
      else{
        SP_AT(tid_y, tid_x) = 0.0;
      }
    }
    else{
      SP_AT(tid_y, tid_x) = 0.0;
    }
    
    __syncthreads();

    //step-5: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
    float val2 = float(SP_AT(tid_y, tid_x));
    val2 = warp_reduce_sum(val2);
    float exp_result = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
    //float exp_result = expf(m_prev[tid_y] - m_new[tid_y]);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val2 = 0.0;//手动实现非并行求rowSum
      for(int h = 0; h < Bc; ++h){
        val2 += SP_AT(tid_y, h);
      }*/
      l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
    }
    __syncthreads();

    //step-6: O = 1/(exp(m_prev - m_new)) * O + P @ V
    if(tid_x < bound_tid_x && tid_y < bound_tid_y){//32路并行计算Oi的每一行
      for(int u = tid_x; u < head_dim; u += blockDim.x){
        float val3 = 0.0;
        #pragma unroll
        for(int w = 0; w < Bc; ++w){//val3 += P[tid_y][w] * V[bid_y][Bc * j + w][bid_x / p][u];
          val3 += float(SP_AT(tid_y, w)) * V_sm_AT(w, u);
        }
        O_sm_AT(tid_y, u) = O_sm_AT(tid_y, u) * exp_result + val3;
      }
    }
    __syncthreads();
      
    //step-7: m_prev <- m_new; l_prev <- l_new
    if (tid_x == 0 && tid_y < bound_tid_y) {//向量更新只使用第1列线程
      m_prev[tid_y] = m_new[tid_y];
      l_prev[tid_y] = l_new[tid_y];
    }
    __syncthreads();

  }
  /****************************end-of-main-loop**************************/

  /*****************************post-process****************************/
  //O(GM) = O/l_prev, aka O_sm /= l_prev and write Oi from SM to GM
  //O[bid_y][Br * bid_z + tid_y][bid_x][*]
  #pragma unroll
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    if(tid_y < bound_tid_y){
      O[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx] = T(O_sm_AT(tid_y, idx) / float(l_prev[tid_y]));
    }
  }
  __syncthreads();
  /*****************************end-of-post-process****************************/

  //取消访问宏定义
  #undef   SP_AT
  #undef   Q_sm_AT
  #undef   K_T_sm_AT
  #undef   V_sm_AT
  #undef   O_sm_AT

}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  //step0: basic check

  //step1: 初始化,预留device端空间
  //cudaStream_t stream2;
  cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, -1);


  const size_t size_bytes_q = h_q.size() * sizeof(T);
  const size_t size_bytes_k = h_k.size() * sizeof(T);
  const size_t size_bytes_v = h_v.size() * sizeof(T);
  const size_t size_bytes_o = h_o.size() * sizeof(T);
  const size_t total_bytes = size_bytes_q + size_bytes_k + size_bytes_v + size_bytes_o;

  //device端只支持裸指针
  T* d_all = nullptr;
  RUNTIME_CHECK(cudaMallocAsync(&d_all, total_bytes, stream2));
  // 切片为Q/K/V/O
  T *d_q = d_all;
  T *d_k = d_q + h_q.size();
  T *d_v = d_k + h_k.size();
  T *d_o = d_v + h_v.size();

  //step2: 拷贝数据from host to device
  RUNTIME_CHECK(cudaMemcpyAsync(d_q, h_q.data(), size_bytes_q, cudaMemcpyHostToDevice, stream2));
  RUNTIME_CHECK(cudaMemcpyAsync(d_k, h_k.data(), size_bytes_k, cudaMemcpyHostToDevice, stream2));
  RUNTIME_CHECK(cudaMemcpyAsync(d_v, h_v.data(), size_bytes_v, cudaMemcpyHostToDevice, stream2));
  RUNTIME_CHECK(cudaMemsetAsync(d_o, 0, size_bytes_o, stream2));//d_o初始化为全0

  //step3: device端计算
  int Br = 32, Bc = 32;
  int grid_dim_z = (target_seq_len + Br - 1) / Br;
  dim3 block_dim(Br, Bc);
  dim3 grid_dim(query_heads, batch_size, grid_dim_z);
  size_t smem_size = (Br * Bc) * sizeof(double) + (Br * 4) * sizeof(float) + (Br * head_dim * 2 + Bc * head_dim * 2) * sizeof(float);

  kernel_flashAttention<T><<<grid_dim, block_dim, smem_size, stream2>>>(batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal, d_q, d_k, d_v, d_o);//注意核函数返回类型只能为void
  RUNTIME_CHECK(cudaStreamSynchronize(stream2));//important

  //step4: 拷贝数据from device to host
  RUNTIME_CHECK(cudaMemcpyAsync(h_o.data(), d_o, size_bytes_o, cudaMemcpyDeviceToHost, stream2));
  RUNTIME_CHECK(cudaStreamSynchronize(stream2));//important

  //step5: free memory
  RUNTIME_CHECK(cudaFreeAsync(d_all, stream2));

  //std::cout << "h_o[0] is: " << float(h_o[0]) << std::endl;
  return;
}
#endif





#ifdef PLATFORM_ILUVATAR
class myDouble{
private:
    float _hi;
    float _lo;

public:
//Big 3
    __host__ __device__
    myDouble(float hi = 0.0f, float lo = 0.0f): _hi(hi), _lo(lo){}//显式初始化

    __host__ __device__
    myDouble(const myDouble& other){//拷贝初始化
        _hi = other._hi;
        _lo = other._lo;
    }

    __host__ __device__
    ~myDouble(){}//析构

    __host__ __device__
    myDouble& operator=(const myDouble& other){//拷贝赋值
        if(this != &other){
            _hi = other._hi;
            _lo = other._lo;
        }
        return *this;
    }


//auxiliary functions
    __host__ __device__
    myDouble operator*(const float op) const {//myDuoble * float
        float p_hi = _hi * op;
        float p_lo = __fmaf_rn(_hi, op, -p_hi); // 捕捉 hi 乘法的剩余误差
        p_lo += (_lo * op);                    // 累加 lo 部分的乘积
        float final_hi = p_hi + p_lo;
        float final_lo = p_lo - (final_hi - p_hi);

        return myDouble(final_hi, final_lo);
    }

    
    __host__ __device__
    myDouble operator-(const float op) const {//myDouble - float
        float s_hi = _hi - op;
        float v = s_hi - _hi;
        float err = (-op) - (v + (s_hi - v - _hi)); 
  
        float s_lo = _lo + err;
        float final_hi = s_hi + s_lo;
        float final_lo = s_lo - (final_hi - s_hi);

        return myDouble(final_hi, final_lo);
    }
    

    __host__ __device__
    myDouble exp(void) const{
        float final_hi = expf(_hi) * (1.0f + _lo);
        float final_lo = 0.0f;
        return myDouble(final_hi, final_lo);
    }

    __host__ __device__
    myDouble exp1(void) const {
        float p_hi = expf(_hi);
        float p_lo = p_hi * _lo;

        float final_hi = p_hi + p_lo;
        float final_lo = p_lo - (final_hi - p_hi);
        return myDouble(final_hi, final_lo);
    }


//visit functions
    __host__ __device__
    float hi(void) const{
        return _hi;
    }

    __host__ __device__
    float lo(void) const{
        return _lo;
    }

    __host__ __device__
    float asfloat(void) const{
        return _hi + _lo;
    }

};



__device__ myDouble mylut(int head_dim){
    switch(head_dim){
        case 1: return myDouble(1.0f, 0.0f);                    break; //0x3ff0000000000000
        case 2: return myDouble(0.707106769f, 1.21016172e-8f);  break; //0x3fe6a09e667f3bcc
        case 4: return myDouble(0.5f, 0.0f);                    break; //0x3fe0000000000000
        case 8: return myDouble(0.353553385f, 5.5932738e-9f);   break; //0x3fd6a09e667f3bcc
        case 16:return myDouble(0.25f, 0.0f);                   break; //0x3fd0000000000000
        case 32:return myDouble(0.176776695f, 2.96636886e-10f); break; //0x3fc6a09e667f3bcc
        case 64:return myDouble(0.125f, 0.0f);                  break; //0x3fc0000000000000
        default:return myDouble(0.0f, 0.0f);                    break;
    }
}


//对应flash attention v2原文算法, block采用三维布局
template <typename T>
__global__ void kernel_flashAttention(int batch_size, int target_seq_len, int src_seq_len, int q_heads, int kv_heads, int head_dim, bool is_causal, const T* Q, const T* K, const T* V, T* O){
  int tid_x = threadIdx.x;//横向,blockDim.x列
  int tid_y = threadIdx.y;//纵向,blockDim.y行
  int bid_x = blockIdx.x;//x方向,总数 = #q_heads
  int bid_y = blockIdx.y;//y方向,总数 = #batch
  int bid_z = blockIdx.z;//z方向,总数 = Tr
  const int p = q_heads / kv_heads;//计算比例系数
  const int Br = blockDim.y;//Q纵向每块大小, 默认为32 (RTX 5090)
  const int Bc = blockDim.x;//K/V纵向分块大小, 默认为32
  const int Tc = (src_seq_len + Bc - 1) / Bc;//对应原始论文中K/V纵向分块数Tc,其中Bc = 32

  //预计算常量
  const float scale_factor_f = 1.0 / sqrt(float(head_dim));//保留精度,采用double
  const myDouble scale_factor = mylut(head_dim);//保留精度,采用double


  //定义一系列临时变量
  /*__shared__ double SP[Br][Bc];//复用S和P
  __shared__ double m_prev[Br], m_new[Br];
  __shared__ double l_prev[Br], l_new[Br];

  __shared__ float Q_sm[Br][64];
  __shared__ float K_T_sm[64][Bc];//transpose for K
  __shared__ float V_sm[Bc][64];
  __shared__ float O_sm[Br][64];*/

  extern __shared__ char shared_mem[];
  char* ptr = shared_mem;  
  //计算中间变量,包括S, P(复用为SP), m_prev, m_new, l_prev, l_new; 为保留精度, SP采用double
  myDouble* SP = reinterpret_cast<myDouble*>(ptr);    // double SP[Br][Bc]
  ptr += Br * Bc * sizeof(myDouble);
  float* m_prev = reinterpret_cast<float*>(ptr);  // float m_prev[Br]
  ptr += Br * sizeof(float);
  float* m_new = reinterpret_cast<float*>(ptr);   // float m_new[Br] 
  ptr += Br * sizeof(float);
  float* l_prev = reinterpret_cast<float*>(ptr);  // float l_prev[Br]
  ptr += Br * sizeof(float);
  float* l_new = reinterpret_cast<float*>(ptr);   // float l_new[Br] 
  ptr += Br * sizeof(float);  

  //原始数据QKV和计算结果O; 全采用float
  float* Q_sm = reinterpret_cast<float*>(ptr);    // float Q_sm[Br][head_dim] 
  ptr += Br * head_dim * sizeof(float);  
  float* K_T_sm = reinterpret_cast<float*>(ptr);  // float K_T_sm[head_dim][Bc]
  ptr += head_dim * Bc * sizeof(float);
  float* V_sm = reinterpret_cast<float*>(ptr);    // float V_sm[Br][head_dim] 
  ptr += Bc * head_dim * sizeof(float);  
  float* O_sm = reinterpret_cast<float*>(ptr);    // float O_sm[Br][head_dim]

  //定义访问宏
  /*#define   SP_AT(y, x)       SP[y][x]
  #define   Q_sm_AT(y, x)     Q_sm[y][x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y][x]
  #define   V_sm_AT(y, x)     V_sm[y][x]
  #define   O_sm_AT(y, x)     O_sm[y][x]*/

  #define   SP_AT(y, x)       SP[y * Bc + x]
  #define   Q_sm_AT(y, x)     Q_sm[y * head_dim + x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y * Bc + x]
  #define   V_sm_AT(y, x)     V_sm[y * head_dim + x]
  #define   O_sm_AT(y, x)     O_sm[y * head_dim + x]


  /****************************preparation**************************/
  int bound_tid_y = ::min(Br, target_seq_len - Br * bid_z);

  //preparation-1: load Qi from GM to SM, and reset Oi to 0
  //Q[bid_y][Br * bid_z + tid_y][bid_x][*]
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    O_sm_AT(tid_y, idx) = 0.0;
    Q_sm_AT(tid_y, idx) = 0.0;
    if(tid_y < bound_tid_y){
      Q_sm_AT(tid_y, idx) = float(Q[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx]);
    }
  }
  __syncthreads();

  //preparation-2: reset m_prev to -INFINITY and l_prev to 0
  if(tid_x == 0){
    m_prev[tid_y] = -8192.0;
    l_prev[tid_y] = 0.0;
  }
  __syncthreads();
  /****************************end-of-preparation*************************/


  /****************************main-loop**************************/
  for(int j = 0; j < Tc; ++j){//对于每个K/V分块
    bool skip = (is_causal && bid_z < j);
    if(skip){//early exit, 直接跳过
    __syncthreads();
      continue;
    }

    SP_AT(tid_y, tid_x) = myDouble(-8192.0f, 0.0f);
    __syncthreads();
    int bound_tid_x = ::min(Bc, src_seq_len - Bc * j);
    bool is_compute = true;//optimization: 分支处理,加速branch-resolving
    if (is_causal) {
      if (bid_z < j) {
        is_compute = false;  // 早期退出情况
      } else if (bid_z == j) {
        is_compute = (tid_y >= tid_x);  // 对角线以上
      }
    }

    //step-1: load Ki, Vi from GM to SM, reset Oi to 0
    //K[bid_y][Bc * j + tid_y][bid_x / p][*], V[bid_y][Bc * j + tid_y][bid_x / p][*]
    for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
      K_T_sm_AT(idx, tid_y) = 0.0;
      V_sm_AT(tid_y, idx) = 0.0;
      if(tid_y < bound_tid_x){//注意这里是bound_tid_x
        K_T_sm_AT(idx, tid_y) = float(K[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
        V_sm_AT(tid_y, idx) = float(V[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
      }
    }
    __syncthreads();

    //step-2: S = Q @ K.T, point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){//用于边缘不完整块
      float val0 = 0.0;//临时sum
      if(is_compute){
        for(int k = 0; k < head_dim; ++k){
          val0 += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, tid_x);
        }
        //SP_AT(tid_y, tid_x) = val0 * scale_factor;//必须用double,对精度影响最大的计算步骤
        SP_AT(tid_y, tid_x) = scale_factor * val0;//myDouble * float
        //SP_AT(tid_y, tid_x) = myDouble(scale_factor_f * val0, 0.0f);//myDouble * float
      }
    }
    __syncthreads();

    //step-3: m_new = max(m_prev, rowMax(S))
    float val1 = SP_AT(tid_y, tid_x).asfloat();
    val1 = warp_reduce_max(val1);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val1 = SP_AT(tid_y, 0);//手动实现非并行求行最大值
      for(int h = 1; h < Bc; ++h){
        val1 = (val1 < SP_AT(tid_y, h)) ? SP_AT(tid_y, h) : val1;
      }*/
      m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
    }
    __syncthreads();

    //step-4: P = exp(S - m_new), point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){
      if(is_compute){
        //SP_AT(tid_y, tid_x) = myexp<float>(SP_AT(tid_y, tid_x) - m_new[tid_y]);
        myDouble tmp = SP_AT(tid_y, tid_x) - m_new[tid_y];//myDouble - float
        SP_AT(tid_y, tid_x) = tmp.exp();
      }
      else{
        SP_AT(tid_y, tid_x) = myDouble(0.0f, 0.0f);
      }
    }
    else{
      SP_AT(tid_y, tid_x) = myDouble(0.0f, 0.0f);
    }
    
    __syncthreads();

    //step-5: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
    float val2 = SP_AT(tid_y, tid_x).asfloat();
    val2 = warp_reduce_sum(val2);
    float exp_result = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
    //float exp_result = expf(m_prev[tid_y] - m_new[tid_y]);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val2 = 0.0;//手动实现非并行求rowSum
      for(int h = 0; h < Bc; ++h){
        val2 += SP_AT(tid_y, h);
      }*/
      l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
    }
    __syncthreads();

    //step-6: O = 1/(exp(m_prev - m_new)) * O + P @ V
    if(tid_x < bound_tid_x && tid_y < bound_tid_y){//32路并行计算Oi的每一行
      for(int u = tid_x; u < head_dim; u += blockDim.x){
        float val3 = 0.0;
        for(int w = 0; w < Bc; ++w){//val3 += P[tid_y][w] * V[bid_y][Bc * j + w][bid_x / p][u];
          val3 += SP_AT(tid_y, w).asfloat() * V_sm_AT(w, u);
        }
        O_sm_AT(tid_y, u) = O_sm_AT(tid_y, u) * exp_result + val3;
      }
    }
    __syncthreads();
      
    //step-7: m_prev <- m_new; l_prev <- l_new
    if (tid_x == 0 && tid_y < bound_tid_y) {//向量更新只使用第1列线程
      m_prev[tid_y] = m_new[tid_y];
      l_prev[tid_y] = l_new[tid_y];
    }
    __syncthreads();

  }
  /****************************end-of-main-loop**************************/

  /*****************************post-process****************************/
  //O(GM) = O/l_prev, aka O_sm /= l_prev and write Oi from SM to GM
  //O[bid_y][Br * bid_z + tid_y][bid_x][*]
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    if(tid_y < bound_tid_y){
      O[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx] = T(O_sm_AT(tid_y, idx) / float(l_prev[tid_y]));
    }
  }
  __syncthreads();
  /*****************************end-of-post-process****************************/

  //取消访问宏定义
  #undef   SP_AT
  #undef   Q_sm_AT
  #undef   K_T_sm_AT
  #undef   V_sm_AT
  #undef   O_sm_AT

}



//iluvatar, 去除Aync资源分配
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  //step0: basic check

  //step1: 初始化,预留device端空间
  const size_t size_bytes_q = h_q.size() * sizeof(T);
  const size_t size_bytes_k = h_k.size() * sizeof(T);
  const size_t size_bytes_v = h_v.size() * sizeof(T);
  const size_t size_bytes_o = h_o.size() * sizeof(T);
  //const size_t size_bytes_lm = target_seq_len * query_heads * batch_size * sizeof(T);
  T *d_q, *d_k, *d_v, *d_o;//device端只支持裸指针
  RUNTIME_CHECK(cudaMalloc(&d_q, size_bytes_q));
  RUNTIME_CHECK(cudaMalloc(&d_k, size_bytes_k));
  RUNTIME_CHECK(cudaMalloc(&d_v, size_bytes_v));
  RUNTIME_CHECK(cudaMalloc(&d_o, size_bytes_o));

  //step2: 拷贝数据from host to device
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), size_bytes_q, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), size_bytes_k, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), size_bytes_v, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_o, 0, size_bytes_o));//d_o初始化为全0

  //step3: device端计算
  int Br = 32, Bc = 32;
  int grid_dim_z = (target_seq_len + Br - 1) / Br;
  dim3 block_dim(Br, Bc);
  dim3 grid_dim(query_heads, batch_size, grid_dim_z);
  size_t smem_size = (Br * Bc) * sizeof(myDouble) + (Br * 4) * sizeof(float) + (Br * head_dim * 2 + Bc * head_dim * 2) * sizeof(float);
  
  kernel_flashAttention<T><<<grid_dim, block_dim, smem_size>>>(batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal, d_q, d_k, d_v, d_o);
  //注意核函数返回类型只能为void

  //step4: 拷贝数据from device to host(not needed)
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, size_bytes_o, cudaMemcpyDeviceToHost));

  //step5: free memory
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));

  //std::cout << "h_o[0] is: " << float(h_o[0]) << std::endl;
  return;
}
#endif





// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
