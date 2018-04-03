#include "lab1.h"
#include "math.h"
#define PI 3.14159265
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 720;

struct Lab1VideoGenerator::Impl {
	int t = 0;
	uint8_t *output;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
	cudaMalloc((void **) &impl->output, sizeof(uint8_t)*W*H);

	uint8_t *tmp = (uint8_t*)malloc(sizeof(uint8_t)*W*H);
	memset(tmp, 0, sizeof(uint8_t)*W*H);

	//printf("00000\n");
	
	/*
	for(int i=0;i<W;i++){
		int ix = int(i/PI*(int(W/2)-1)+int(W/2));
		int iy = int(-std::sin(i)*(int(H/2)-1)+int(H/2));
		tmp[iy*W+ix] = 1;
	}
	*/
	
	cudaMemcpy(impl->output, tmp, sizeof(uint8_t)*W*H, cudaMemcpyHostToDevice);
	free(tmp);


	//printf("22222\n");
}

Lab1VideoGenerator::~Lab1VideoGenerator(){
	//cudaFree(impl->output);
}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__global__ void draw(uint8_t *frame, uint8_t *output, int t){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(y < H && x < W){
		//if(y-sin(x*PI/180)<0.01) frame[y*W+x]=255;
		//if(y<((H-1)/2+1) && x<((W-1)/2+1)) frame[y*W+x]=255;
		//else frame[y*W+x]=0;
		//frame[y*W+x]=255;
		//int t = impl->t;
		if(t%10==0){
			int tt = 7*t*t*t-4*t*t+5*t+13;
			//uint8_t *tmp = (uint8_t*)malloc(sizeof(uint8_t)*W*H);
			for(int i=0;i<int(H);i++){
				for(int j=0;j<int(W);j++){
					int k = int((i*j*tt)%256);
					if(k>128) output[i*W+j] = 255;
					else output[i*W+j]=0;
				}
			}
		}

		frame[y*W+x] = output[y*W+x];
		//frame[y*W+x] = 0;
		//free(tmp);
	}
}


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv, 0, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
	draw<<<dim3( (W-1)/16+1, (H-1)/12+1),dim3(16,12)>>>(yuv, impl->output, impl->t);
	++(impl->t);
}





