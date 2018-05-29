#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if(yt < ht and xt < wt and mask[curt] > 127.0f){
		const int yb = oy+yt, xb = ox+xt;
		if(0 <= yb and yb < hb and 0 <= xb and xb < wb){
			fixed[curt*3+0] = 0;
			fixed[curt*3+1] = 0;
			fixed[curt*3+2] = 0;
			if(yb+1 < hb){ // down
				int neit = wt*(yt+1)+xt;
				if(yt+1<ht){ // in
					fixed[curt*3+0] += target[curt*3+0]-target[neit*3+0];
					fixed[curt*3+1] += target[curt*3+1]-target[neit*3+1];
					fixed[curt*3+2] += target[curt*3+2]-target[neit*3+2];
				}
				if(!(yt+1<ht) || mask[neit]<=127.0f){ //boundary
					int neib = wb*(yb+1)+xb;
					fixed[curt*3+0] += background[neib*3+0];
					fixed[curt*3+1] += background[neib*3+1];
					fixed[curt*3+2] += background[neib*3+2];
				}
			}
			if(yb-1 >= 0){ // up
				int neit = wt*(yt-1)+xt;
				if(yt-1>=0){ // in
					fixed[curt*3+0] += target[curt*3+0]-target[neit*3+0];
					fixed[curt*3+1] += target[curt*3+1]-target[neit*3+1];
					fixed[curt*3+2] += target[curt*3+2]-target[neit*3+2];
				}
				if(!(yt-1>=0) || mask[neit]<=127.0f){ //boundary
					int neib = wb*(yb-1)+xb;
					fixed[curt*3+0] += background[neib*3+0];
					fixed[curt*3+1] += background[neib*3+1];
					fixed[curt*3+2] += background[neib*3+2];
				}
			}
			if(xb+1 < wb){ // right
				int neit = wt*yt+xt+1;
				if(xt+1<wt){ // in
					fixed[curt*3+0] += target[curt*3+0]-target[neit*3+0];
					fixed[curt*3+1] += target[curt*3+1]-target[neit*3+1];
					fixed[curt*3+2] += target[curt*3+2]-target[neit*3+2];
				}
				if(!(xt+1<wt) || mask[neit]<=127.0f){ //boundary
					int neib = wb*yb+xb+1;
					fixed[curt*3+0] += background[neib*3+0];
					fixed[curt*3+1] += background[neib*3+1];
					fixed[curt*3+2] += background[neib*3+2];
				}
			}
			if(xb-1 >= 0){ // left
				int neit = wt*yt+xt-1;
				if(xt-1>=0){ // in
					fixed[curt*3+0] += target[curt*3+0]-target[neit*3+0];
					fixed[curt*3+1] += target[curt*3+1]-target[neit*3+1];
					fixed[curt*3+2] += target[curt*3+2]-target[neit*3+2];
				}
				if(!(xt-1>=0) || mask[neit]<=127.0f){ //boundary
					int neib = wb*yb+xb-1;
					fixed[curt*3+0] += background[neib*3+0];
					fixed[curt*3+1] += background[neib*3+1];
					fixed[curt*3+2] += background[neib*3+2];
				}
			}
		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	const float *target,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if(yt < ht and xt < wt and mask[curt] > 127.0f){
		const int yb = oy+yt, xb = ox+xt;
		if(0 <= yb and yb < hb and 0 <= xb and xb < wb){
			int N = 0;
			float v[3];
			v[0] = fixed[curt*3+0];
			v[1] = fixed[curt*3+1];
			v[2] = fixed[curt*3+2];
			if(yb+1 < hb){ // down
				int neit = wt*(yt+1)+xt;
				if(yt+1<ht && mask[neit]>127.0f){ // in
					v[0] += target[neit*3+0];
					v[1] += target[neit*3+1];
					v[2] += target[neit*3+2];
				}
					N++;
			}
			if(yb-1 >= 0){ // up
				int neit = wt*(yt-1)+xt;
				if(yt-1>=0 && mask[neit]>127.0f){ // in
					v[0] += target[neit*3+0];
					v[1] += target[neit*3+1];
					v[2] += target[neit*3+2];
				}
					N++;
			}
			if(xb+1 < wb){ // right
				int neit = wt*yt+xt+1;
				if(xt+1<wt && mask[neit]>127.0f){ // in
					v[0] += target[neit*3+0];
					v[1] += target[neit*3+1];
					v[2] += target[neit*3+2];
				}
					N++;
			}
			if(xb-1 >= 0){ // left
				int neit = wt*yt+xt-1;
				if(xt-1>=0 && mask[neit]>127.0f){ // in
					v[0] += target[neit*3+0];
					v[1] += target[neit*3+1];
					v[2] += target[neit*3+2];
				}
					N++;
			}
			output[curt*3+0] = v[0]/N;
			output[curt*3+1] = v[1]/N;
			output[curt*3+2] = v[2]/N;
		}
	}
}




void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	//set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaDeviceSynchronize();
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	
	// iterate
	for(int i=0;i<10000;++i){
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2,
			wb, hb, wt, ht, oy, ox
		);
		cudaDeviceSynchronize();
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1,
			wb, hb, wt, ht, oy, ox
		);
		cudaDeviceSynchronize();
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	cudaDeviceSynchronize();

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}










