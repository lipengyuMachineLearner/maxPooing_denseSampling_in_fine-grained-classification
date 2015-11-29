#pragma once

#include"opencv.h"
class CSamplingKernel
{
private:
	int label_;
	int index_;
	string name_src_;

	int width_;
	int height_;
	int channel_;

	Mat data_;
public:
	CSamplingKernel(int width, int height);
	CSamplingKernel(){ ; }

	bool getParam(int &width, int &height);
	bool getData(Mat &data);
	
	void setIndexAndLabel(int label, int index);
	void setSrcName(string name_src);
	void setMask(Mat &mask);
	void setData(IplImage *img, Point center);
	
	float convolution(IplImage *img, Mat &img_convolution, bool sign_expand_kernel, int conv_step);

	void save(string fileName);
	void saveAsImg(string fileName);
	void load(string fileName);


};