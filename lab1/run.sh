nvcc -std=c++11 -I ../utils/ main.cu lab1.cu -o test
./test
avconv -i result.y4m result.mkv

