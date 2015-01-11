#include <opencv2\opencv.hpp>
#include "LightingEstimation.h"
using namespace cv;

int main(void)
{
	Mat src = imread( "../input/input20.jpg", 0 );
    if( !src.data ){
		std::cout<<"Can not load image!"<<std::endl;
        return -1; 
    }
	imshow("source", src);

	LightingEstimation demo;
	Mat res = demo.makeShading(src);
	imshow("shading", res);

    waitKey();

	return 0;
}