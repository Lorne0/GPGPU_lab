#include "lab1.h"
#include "math.h"
#include "time.h"
#define PI 3.14159265
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 840;

struct Lab1VideoGenerator::Impl { int t = 0; };
Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {}
Lab1VideoGenerator::~Lab1VideoGenerator(){}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__device__ void set_color(uint8_t *frame, const int x, const int y, char color ){
	int R, G, B;
	if(color=='r'){R=255, G=0, B=0;}
	else if(color=='g'){R=0, G=255, B=0;}
	else if(color=='b'){R=0, G=0, B=255;}
	else if(color=='y'){R=255, G=255, B=0;}
	else if(color=='k'){R=0, G=0, B=0;}
	else if(color=='w'){R=255, G=255, B=255;}

	frame[y*W+x] = uint8_t(0.299*R + 0.587*G + 0.114*B);
	frame[y/2*W/2+x/2+W*H] = uint8_t(-0.169*R - 0.331*G + 0.500*B + 128);
	frame[y/2*W/2+x/2+W*H+W*H/4] = uint8_t(0.500*R - 0.419*G - 0.081*B + 128);
}

__device__ void set_random_color(uint8_t *frame, const int x, const int y, int base, int rand_c ){
	int R = (base+rand_c)%255;
	int G = (base*2-rand_c)%255;
	int B = (base*base*rand_c)%255;

	frame[y*W+x] = uint8_t(0.299*R + 0.587*G + 0.114*B);
	frame[y/2*W/2+x/2+W*H] = uint8_t(-0.169*R - 0.331*G + 0.500*B + 128);
	frame[y/2*W/2+x/2+W*H+W*H/4] = uint8_t(0.500*R - 0.419*G - 0.081*B + 128);
}

__device__ double _asin(double a){return asin(a)*180/PI;}
__device__ double _acos(double a){return acos(a)*180/PI;}
__device__ double _sqrt(double a){return sqrt(a);}
__device__ double _abs(double a){return abs(a);}
//__device__ void _printxy(int a, int b){printf("x:%d, y:%d\n", a,b);}
//__device__ void _print(int a){printf("%d\n", a);}

__global__ void draw(uint8_t *frame, int tt, int rand_c){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(y < H && x < W){
		int t;
		//part1 before deadline last 10s, nervous
		if(tt<240) t = tt/24+43200-10;
		//part2 close light
		else if(tt>=240 && tt<288) t=0;
		//part3 reverse, ranbow color?
		else if(tt>=288 && tt<432){
			t = tt-288;
			t = t*t*25;
		}
		else if(tt>=432 && tt<576){
			t = 576-tt;
			t = t*t*25;
		}
		//part4 normal clock
		else t=(tt-576)/24;
		//t = tt*_sqrt(double(tt))/5;
		int a=W/2, b=H/2, r=H/2; 
		int rr = (x-a)*(x-a)+(y-b)*(y-b);
		double r2 = _sqrt(double(rr));
		double theta=0, angle=0, theta1=0, theta2=0;
		if(tt<240 || (tt>=432)){theta = 90-(t%60)*6, theta1 = 90-(t%3600)*0.1; theta2 = 90-(t%43200)*0.1/12;}
		else if(tt>=288 && tt<432){theta = 90+(t%60)*6, theta1 = 90+(t%3600)*0.1; theta2 = 90+(t%43200)*0.1/12;}
		if(theta>360) theta-=360;
		if(theta<0) theta+=360;
		if(theta1>360) theta1-=360;
		if(theta1<0) theta1+=360;
		if(theta2>360) theta2-=360;
		if(theta2<0) theta2+=360;
		if((x-a)>0) angle = 360+_asin(-double(y-b)/r2);
		else angle = 180-_asin(-double(y-b)/r2);
		if(angle>360) angle-=360;
		if(angle<0) angle+=360;

		if(rr<=r*r && rr>=(r-3)*(r-3) ){
			if(tt<240){
				set_color(frame, x, y, 'r');
				frame[y*W+x] += 150;
			}
			else if(tt>=288 && tt<576){
				//set_color(frame, x, y, 'w');
				set_random_color(frame, x, y, 123, rand_c);
			}
			else if(tt>=576){
				set_color(frame, x, y, 'b');
				frame[y*W+x] += 210;
			}
		}
		if(rr<=r*r){
			if(int(angle)%6==0 && rr>=(r-3)*(r-3)) set_color(frame, x, y, 'k');
			if(int(angle)%30==0 && rr>=(r-3)*(r-3)) set_color(frame, x, y, 'g');
			if(_abs(angle-theta)<0.5 && rr<(r-3)*(r-3)){
				if(tt<240){
					set_color(frame, x, y, 'r');
					frame[y*W+x] += 50;
				}
				else if(tt>=288 && tt<576){
					//set_color(frame, x, y, 'w');
					set_random_color(frame, x, y, 77, rand_c);
				}
				else if(tt>=576){
					set_color(frame, x, y, 'b');
					frame[y*W+x] += 150;
				}
			}
			
		}
		if(rr<=(r-5)*(r-5)){
			if(_abs(angle-theta1)<1){
				if(tt<240){
					set_color(frame, x, y, 'r');
					frame[y*W+x] -= 10;
				}
				else if(tt>=288 && tt<576){
					//set_color(frame, x, y, 'w');
					set_random_color(frame, x, y, 13, rand_c);
				}
				else if(tt>=576){
					set_color(frame, x, y, 'b');
					frame[y*W+x] += 70;
				}
			}
		}
		if(rr<=(r-100)*(r-100)){
			if(_abs(angle-theta2)<2){
				if(tt<240){
					set_color(frame, x, y, 'r');
					frame[y*W+x] -= 70;
				}
				else if(tt>=288 && tt<576){
					//set_color(frame, x, y, 'w');
					set_random_color(frame, x, y, 168, rand_c);
				}
				else if(tt>=576){
					set_color(frame, x, y, 'b');
					frame[y*W+x] -= 20;
				}
			}
		}


	}
}


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv, 0, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
	//cudaMemset(yuv+W*H, 0, W*H/4);
	//cudaMemset(yuv+W*H+W*H/4, 149, W*H/4);
	//uint8_t* sky = (uint8_t*)malloc(sizeof(uint8_t)*W*H*2);
	//draw<<<dim3( (W-1)/16+1, (H-1)/12+1),dim3(16,12)>>>(yuv, sky, stars, impl->t);
	//unsigned seed = (unsigned)time(NULL);
	//srand(seed);
	int r = rand()%255;
	
	draw<<<dim3( (W-1)/16+1, (H-1)/12+1),dim3(16,12)>>>(yuv, impl->t, r);
	++(impl->t);
}





