#include "LightingEstimation_marker.h"
#include <fstream>
#include <time.h>
using namespace cv;
using namespace std;


float dsFactor = 1.0f;        //downsample ratio
double output[8],cost;
ofstream fout;

struct onMouseData
{
	Mat *img;
	std::vector<Point2f> srcPts;
	std::vector<Point2f> desPts;
};

static void onMouse( int event, int x, int y, int, void* d )
{
	static int count = 0;
	onMouseData *data = static_cast<onMouseData *>(d);
	if(event != EVENT_LBUTTONDOWN)
		return;
	
	if(count<4){
		Mat click_img = data->img->clone();
		data->srcPts[count++] = Point2f((float)x*dsFactor, (float)y*dsFactor);
		circle(click_img, Point2f((float)x, (float)y), 5, Scalar(0,0,255));
		imshow("img", click_img);

		std::cout<<"("<<data->srcPts[count-1].x<<","<<data->srcPts[count-1].y<<")";
		std::cout<<" --> ("<<data->desPts[count-1].x<<","<<data->desPts[count-1].y<<")";
		std::cout<<std::endl;
	}
	if(count==4){
		destroyWindow("img");
		count++;
		resize(*(data->img), *(data->img), Size((int)(data->img->cols*dsFactor), (int)(data->img->rows*dsFactor)));
		Mat H = getPerspectiveTransform(data->srcPts, data->desPts);    //find homography

		clock_t start, end;
		start = clock();
		cost = LE_marker::getInstance().estimate(*(data->img), H);
		end = clock();
		cout<<"execution time:\t"<<(end-start)<<" ms"<<endl;
		
		LE_marker::getInstance().outputData(output);
		fout<<"ambient = "<<output[0]<<endl;
		fout<<"diffuse = "<<output[1]<<endl;
		fout<<"normal = ("<<output[2]<<", "<<output[3]<<", "<<output[4]<<")"<<endl;
		fout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
		fout<<"minimum cost = "<<cost<<endl;
		fout<<"================================="<<endl;
		fout<<"execution time:\t"<<static_cast<float>(end-start)/1000.0f<<" s"<<endl;

		system("cls");
		cout<<"Finished!"<<endl;
		cout<<"execution time:\t"<<static_cast<float>(end-start)/1000.0f<<" s"<<endl;
		cout<<"================================="<<endl;
		cout<<"ambient = "<<output[0]<<endl;
		cout<<"diffuse = "<<output[1]<<endl;
		cout<<"normal = ("<<output[2]<<", "<<output[3]<<", "<<output[4]<<")"<<endl;
		cout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
		cout<<"minimum cost = "<<cost<<endl;
	}
}

int main(int argc, char* argv[])
{
	/*
		參數argv[1] input檔案路徑
		   argv[2] output檔案名
	*/


	string imgPath(argv[1]);
	Mat img = imread(imgPath);
	imshow("img", img);

	fout.open(argv[2]);
	fout<<"name : "<<imgPath<<endl;

	onMouseData data;
	data.srcPts.resize(4);
	data.desPts.resize(4);

	data.img = &img;

	/*手點*/
	float half_markerLen = 20.0/2;
	data.desPts[0] = Point2f(-half_markerLen,  half_markerLen);            //clockwise
	data.desPts[1] = Point2f(half_markerLen, half_markerLen);
	data.desPts[2] = Point2f( half_markerLen, -half_markerLen);
	data.desPts[3] = Point2f( -half_markerLen,  -half_markerLen);
	/***/

	//double w = img.cols, h = img.rows;
	//data.srcPts[0] = Point2f(0.0f, 0.0f);
	//data.srcPts[1] = Point2f(w, 0.0f);
	//data.srcPts[2] = Point2f(w, h);
	//data.srcPts[3] = Point2f(0.0f, h);

	//data.desPts[0] = Point2f(-(w/2), (h/2));
	//data.desPts[1] = Point2f((w/2), (h/2));
	//data.desPts[2] = Point2f((w/2), -(h/2));
	//data.desPts[3] = Point2f(-(w/2), -(h/2));
	//
	//onMouseData *ptr = &data;
	//resize(*(ptr->img), *(ptr->img), Size((int)(ptr->img->cols*dsFactor), (int)(ptr->img->rows*dsFactor)));
	//Mat H = getPerspectiveTransform(data.srcPts, data.desPts);    //find homography

	if(argc > 3){
		double initGuess[8];
		for(int i=0;i<8;i++)
			initGuess[i] = atof(argv[i+3]);
		LE_marker::getInstance().setInitGuess(initGuess[0], initGuess[1],
			initGuess[2], initGuess[3], initGuess[4],
			initGuess[5], initGuess[6], initGuess[7]);
		fout<<"initial guess: ["<<initGuess[0]<<", "<<initGuess[1]<<", ";
		fout<<initGuess[2]<<", "<<initGuess[3]<<", "<<initGuess[4]<<", ";
		fout<<initGuess[5]<<", "<<initGuess[6]<<", "<<initGuess[7]<<"]"<<endl;
	}
	else{
		double initGuess[] = {0.15, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0};
		LE_marker::getInstance().setInitGuess(initGuess[0], initGuess[1],
			initGuess[2], initGuess[3], initGuess[4],
			initGuess[5], initGuess[6], initGuess[7]);
		fout<<"initial guess: ["<<initGuess[0]<<", "<<initGuess[1]<<", ";
		fout<<initGuess[2]<<", "<<initGuess[3]<<", "<<initGuess[4]<<", ";
		fout<<initGuess[5]<<", "<<initGuess[6]<<", "<<initGuess[7]<<"]"<<endl;
	}

	/*clock_t start, end;
	start = clock();
	cost = LE_marker::getInstance().estimate(*(ptr->img), H);
	end = clock();
	cout<<"excution time:\t"<<(end-start)<<" ms"<<endl;*/

	/*LE_marker::getInstance().outputData(output);
	fout<<"ambient = "<<output[0]<<endl;
	fout<<"diffuse = "<<output[1]<<endl;
	fout<<"normal = ("<<output[2]<<", "<<output[3]<<", "<<output[4]<<")"<<endl;
	fout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
	fout<<"minimum cost = "<<cost<<endl;

	cout<<"\nFinished!"<<endl;
	cout<<"================================="<<endl;
	cout<<"ambient = "<<output[0]<<endl;
	cout<<"diffuse = "<<output[1]<<endl;
	cout<<"normal = ("<<output[2]<<", "<<output[3]<<", "<<output[4]<<")"<<endl;
	cout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
	cout<<"minimum cost = "<<cost<<endl;*/

	
	/*手點*/
	setMouseCallback("img", onMouse, &data);
	/***/	

	waitKey();
	fout.close();
	
	return 0;
}