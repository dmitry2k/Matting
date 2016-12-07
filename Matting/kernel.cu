#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024
#define LOW_PASS 30
#define HIGH_PASS 245

#define CHECK(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
			<< " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
		exit(1);															\
	} }

//удаление зеленного фона по границе
__global__ void greensceen(uchar *a, int n1, int m1, uchar *b, int n2, int m2)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int threadsNum = blockDim.x*gridDim.x;
	for (int i = id; i < n1*m1; i += threadsNum)
	{
		int cur_row = i / m1;
		int cur_col = i % m1;
		int new_i3 = 3 * (m2*cur_row + cur_col);
		int i3 = 3 * i;


		if (!(a[i3 + 1] > a[i3] && a[i3 + 1] > a[i3 + 2] && a[i3 + 1] > LOW_PASS && a[i3 + 1] < HIGH_PASS))
		{
			b[new_i3] = a[i3];
			b[new_i3 + 1] = a[i3 + 1];
			b[new_i3 + 2] = a[i3 + 2];
		}

	}
}

int main(void)
{
	VideoCapture cap("video1.avi"); // видео с объектом на зеленом фоне
	if (!cap.isOpened())
	{
		cout << "Error can't find the file" << endl;
	}
	VideoCapture cap2("video2.avi"); // видео-фон
	if (!cap2.isOpened())
	{
		cout << "Error can't find the file2" << endl;
	}

	Mat frame, frame2;
	namedWindow("1", WINDOW_AUTOSIZE); // окно с объектом на зеленом фоне
	namedWindow("2", WINDOW_AUTOSIZE); // окно с фоном
	namedWindow("3", WINDOW_AUTOSIZE); // результат

	/*
	VideoWriter outputVideo;
	outputVideo.open("video3.avi", cap2.get(CV_CAP_PROP_FOURCC), cap2.get(CV_CAP_PROP_FPS), Size(cap2.get(CV_CAP_PROP_FRAME_WIDTH), cap2.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video file\n";
		return -1;
	}
	*/

	int m1 = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int n1 = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int m2 = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
	int n2 = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);
	uchar *dev_a, *dev_b;

	cudaEvent_t stopCUDA;
	cudaEventCreate(&stopCUDA);

	CHECK(cudaMalloc(&dev_a, 3 * n1*m1 * sizeof(uchar)));
	CHECK(cudaMalloc(&dev_b, 3 * n2*m2 * sizeof(uchar)));

	while (true)
	{
		if (!cap.read(frame)) break;
		if (!cap2.read(frame2)) break;
		imshow("1", frame);
		imshow("2", frame2);
		CHECK(cudaMemcpy(dev_a, frame.data, 3 * n1*m1 * sizeof(uchar), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(dev_b, frame2.data, 3 * n2*m2 * sizeof(uchar), cudaMemcpyHostToDevice));

		greensceen<<<GRID_SIZE, BLOCK_SIZE >>>(dev_a, n1, m1, dev_b, n2, m2);

		cudaEventRecord(stopCUDA, 0);
		cudaEventSynchronize(stopCUDA);

		CHECK(cudaMemcpy(frame2.data, dev_b, 3 * n2*m2 * sizeof(uchar), cudaMemcpyDeviceToHost));
		imshow("3", frame2);
		//outputVideo.write(frame2);
		waitKey(33);
	}

	CHECK(cudaFree(dev_a));
	CHECK(cudaFree(dev_b));

	system("pause");

	return 0;
}
