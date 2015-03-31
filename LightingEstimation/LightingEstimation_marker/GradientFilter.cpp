#include "GradientFilter.h"
#define _LE_DEBUG
using namespace cv;

GradientFilter::GradientFilter()
{
	lambda = 0.1;
	tao = 0.5;
	stopItrCost = 0.01;
	stopMaxItrCount = 1;
	stopPixelVal = 10.0;
}
GradientFilter::~GradientFilter()
{
	delete opt;
}

bool GradientFilter::init(cv::Mat img)
{
	if(img.empty()) 
		return false;
	if(img.channels() != 1){
		cvtColor(img, img, CV_RGB2GRAY);
	}
	imgSize = Size(img.cols, img.rows);
	if(img.cols > img.rows)
		resize(img, img, Size(40,30));
	else if(img.cols == img.rows)
		resize(img, img, Size(35,35));
	else
		resize(img, img, Size(30,40));
	img.convertTo(m_inputImg, CV_64FC1, 1/255.0);	  // uint8 to double
	m_pixelWeights = getPixelWeight(m_inputImg);

	data.lambda = 0.1;
	data.itrCount = 0;
	m_clipped_grad = gradClipping(getGradient(m_inputImg), tao);   //get gradient 
		
	//opt = new nlopt::opt(nlopt::LN_COBYLA, m_inputImg.rows*m_inputImg.cols);    /* algorithm and dimensionality */
	opt = new nlopt::opt(nlopt::LN_COBYLA, 1);    /* algorithm and dimensionality */

	opt->set_min_objective(objFunc, &data);
	opt->set_stopval(stopPixelVal);
	//opt->set_maxtime(120);       /* set stopping criteria*/
	//std::vector<double> lb(m_inputImg.rows*m_inputImg.cols, 1e-8);
	//opt->set_lower_bounds(lb);

	return true;
}

double GradientFilter::objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data)
{
	ObjFunc_data *data =  reinterpret_cast<ObjFunc_data*>(objFunc_data);
	data->patch.at<double>(1,1) = x[0];
	Mat grad_x = getGradient(data->patch);
	double a = grad_x.at<double>(1,1)-(data->clipped_grad_x);
	a = pow(a, 2);
	double b = x[0] - data->inputImg_x;
	b = (data->lambda)*(data->pixelWeight_x)*pow((x[0] - (data->inputImg_x)), 2);

	return a+b;
}

double GradientFilter::constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data)
{
	return 0.0;
}

Mat GradientFilter::optimize()
{  
	double sumCost = 0.0;
	Mat currentImg = m_inputImg.clone();
	do{
		sumCost = 0.0;  
// TODO: OpenCL
		for(int i=1;i<m_inputImg.rows-1;i++){
			for(int j=1;j<m_inputImg.cols-1;j++){
				std::vector<double> x;
				x.push_back(currentImg.at<double>(i,j));
				data.inputImg_x = m_inputImg.at<double>(i,j);
				data.clipped_grad_x = m_clipped_grad.at<double>(i,j);
				data.pixelWeight_x = m_pixelWeights.at<double>(i,j);
				data.patch = currentImg(Range(i-1, i+1), Range(j-1, j+1));

				double cost = 0.0;
				nlopt::result result = opt->optimize(x, cost);
				currentImg.at<double>(i,j) = x[0];
				sumCost += cost;
			}
		}
#ifdef _LE_DEBUG
			std::cout<<"iteration "<<(data.itrCount)+1<<": cost = "<<sumCost<<std::endl;
			/* show progress */
			Mat show;
			resize(currentImg, show, imgSize);
			imshow("Current Image", show);
			waitKey(10);
#endif

	}while (sumCost>=stopItrCost && ++(data.itrCount) < stopMaxItrCount);

	//Mat residual = abs(m_inputImg - currentImg);
	//resize(residual, residual, imgSize);
	//imshow("residual", residual);
	Mat output;
	resize(currentImg, output, imgSize);
	return output;
}

Mat GradientFilter::getGradient(Mat src )
{
    Mat src_gray, grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_64F;

    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	if(src.channels() != 1)
		cvtColor( src, src_gray, CV_RGB2GRAY );
	else
		src_gray = src;
	
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_x, abs_grad_x );
	abs_grad_x = abs(grad_x);
	
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_y, abs_grad_y );
	abs_grad_y = abs(grad_y);

    //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	
	grad = abs_grad_x + abs_grad_y;

	//imshow("gradient", grad);
	return grad;
}

Mat GradientFilter::gradClipping( Mat gradient, double tao )
{
	Mat grad_Clipped, grad_f, clipped_d;

	if(tao == -1.0){
		/* 取平均值作為threshold */
		Mat m1, m2;
		reduce(gradient, m1, 0, CV_REDUCE_AVG);
		reduce(m1, m2, 1, CV_REDUCE_AVG);
		double mean = m2.at<double>(0);
		std::cout<<"The mean of gradient = "<<mean<<std::endl;
		tao = mean;
	}
#ifdef _LE_DEBUG
	std::cout<<"set tao = "<<tao<<std::endl;
#endif
	gradient.convertTo(grad_f, CV_32FC1);
	threshold(grad_f, grad_Clipped, tao, 1.0, THRESH_TOZERO_INV);   //threshold只能接受uint8 or float
	grad_Clipped.convertTo(clipped_d, CV_64FC1);

	//std::cout<<grad_f.row(0);
	//imshow("clipped image", grad_Clipped);
	return clipped_d;

}

Mat GradientFilter::getPixelWeight( Mat img )
{
	Mat img_G, Ones, specH;
	GaussianBlur(img, img_G, Size(101,101), 0, 0, BORDER_DEFAULT);
	Ones = Mat::ones(img.size(), img.type());
	Mat result = Ones - abs(img - img_G);
	specH = Ones.clone();      // zero when the pixel is classified as a specular highlight
	result = result.mul(specH);
	Mat result_double;
	result.convertTo(result_double, CV_64FC1);
	
	//imshow("weight", result_double);
	return result_double;
}