#include "LightingEstimation_marker.h"
#include <fstream>
#include <time.h>
using namespace cv;
using namespace std;


float dsFactor = 1.0f;        //downsample ratio
double output[8],cost;
ofstream fout("output.txt");

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
		count++;
		resize(*(data->img), *(data->img), Size((int)(data->img->cols*dsFactor), (int)(data->img->rows*dsFactor)));
		Mat H = getPerspectiveTransform(data->srcPts, data->desPts);    //find homography

		clock_t start, end;
		start = clock();
		cost = LE_marker::getInstance().estimate(*(data->img), H);
		end = clock();
		cout<<"excution time:\t"<<(end-start)<<" ms"<<endl;
		
		LE_marker::getInstance().outputData(output);
		fout<<"ambient = "<<output[0]<<endl;
		fout<<"diffuse = "<<output[1]<<endl;
		fout<<"normal = ("<<output[2]<<", "<<output[3]<<", "<<output[4]<<")"<<endl;
		fout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
		fout<<"minimum cost = "<<cost<<endl;

		system("cls");
		cout<<"Finished!"<<endl;
		cout<<"================================="<<endl;
		cout<<"ambient = "<<output[0]<<endl;
		cout<<"diffuse = "<<output[1]<<endl;
		cout<<"normal = ("<<output[2]<<", "<<output[3]<<", "<<output[4]<<")"<<endl;
		cout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
		cout<<"minimum cost = "<<cost<<endl;
	}
}

int main(void)
{
	string imgPath = "../input/synth_17.jpg";
	Mat img = imread(imgPath);
	imshow("img", img);
	fout<<"name : "<<imgPath<<endl;

	onMouseData data;
	data.srcPts.resize(4);
	data.desPts.resize(4);

	data.img = &img;
	float half_markerLen = 20.0/2;
	data.desPts[0] = Point2f(-half_markerLen,  half_markerLen);            //clockwise
	data.desPts[1] = Point2f(half_markerLen, half_markerLen);
	data.desPts[2] = Point2f( half_markerLen, -half_markerLen);
	data.desPts[3] = Point2f( -half_markerLen,  -half_markerLen);

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

	double initGuess[] = {0.15, 0.6, -25.0, -25.0, 5.0};
	LE_marker::getInstance().setInitGuess(initGuess[0], initGuess[1], (float)initGuess[2], (float)initGuess[3], (float)initGuess[4]);
	fout<<"initial guess: ["<<initGuess[0]<<", "<<initGuess[1]<<", ";
	fout<<initGuess[2]<<", "<<initGuess[3]<<", "<<initGuess[4]<<"]"<<endl;

	/*clock_t start, end;
	start = clock();
	cost = LE_marker::getInstance().estimate(*(ptr->img), H);
	end = clock();
	cout<<"excution time:\t"<<(end-start)<<" ms"<<endl;
		
	LE_marker::getInstance().outputData(output);
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


	setMouseCallback("img", onMouse, &data);

	waitKey();
	fout.close();
	
	return 0;
}