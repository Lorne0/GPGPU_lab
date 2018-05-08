#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct ZeroOne : public thrust::unary_function<char, int>{
	__host__ __device__ int operator()(char c) { return (c=='\n')? 0:1; }
};

void CountPosition1(const char *text, int *pos, int text_size){
	thrust::device_ptr<const char> dtext(text);
	thrust::device_ptr<int> dpos(pos);
	thrust::transform(dtext, dtext+text_size, dpos, ZeroOne());
	thrust::inclusive_scan_by_key(dpos, dpos+text_size, dpos, dpos);
}

__global__ void InBlock(const char *text, int *pos, int *a, int text_size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int tid = threadIdx.x;
	// change all to 01
	if(idx<text_size){
		pos[idx] = (text[idx] == '\n')? 0:1;
		a[idx] = pos[idx];
	}
	__syncthreads();
	
	
	if(idx<text_size){
		for(int i=0;i<=9;i++){
			int check_len = 1<<i;
			if(tid>=check_len && pos[idx]>=check_len)
				a[idx] += pos[idx-check_len];
			__syncthreads();

			if(tid>=check_len && pos[idx]>=check_len)
				pos[idx] = a[idx];
			__syncthreads();
		}
	}
	
}

__global__ void Block_Merge(const char *text, int *pos, int text_size, int block_size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int tid = threadIdx.x;
	if(blockIdx.x>0){
		if(tid<500 && (tid+1)==pos[idx]){
			pos[idx] += pos[(blockIdx.x-1)*blockDim.x+block_size-1];
		}
	}

}

void CountPosition2(const char *text, int *pos, int text_size){
	int block_size = 1024;
	int block_num = CeilDiv(text_size, block_size);
	int *a;
	cudaMalloc(&a, sizeof(int)*text_size);
	InBlock<<<block_num, block_size>>>(text, pos, a, text_size);
	Block_Merge<<<block_num, block_size>>>(text, pos, text_size, block_size);
}

/*
0 1 1 1
0 1 2 2 
0 1 2 3

0 1 1 1 1 1
0 1 2 2 2 2
0 1 2 3 4 4
0 1 2 3 4 5

*/





