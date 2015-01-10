#include <opencv2\opencv.hpp>
#include "GradientFilter.h"
using namespace cv;

int main(void)
{
	Mat src = imread( "../input/input20.jpg", 0 );
    if( !src.data ){
		std::cout<<"Can not load image!"<<std::endl;
        return -1; 
    }
	imshow("source", src);
///////////////////////////////////////////////////////////////////////////////////
	/* Example of nlopt*/
	GradientFilter demo;
	demo.init(src);
	Mat shading = demo.optimize();
	/*Mat shading_8U;
	shading.convertTo(shading_8U, CV_8UC1, 255);
	resize(shading_8U, shading_8U, Size(src.cols,src.rows));
	imshow("shading", shading_8U);*/

/////////////////////////////////////////////////////////////////////////////////////

    waitKey();

	return 0;
}