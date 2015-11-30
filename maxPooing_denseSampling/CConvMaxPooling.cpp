#include "CConvMaxPooling.h"

CConvMaxPooling::CConvMaxPooling(int num_kernel, int num_img, vector<string> &img_files, vector<int> &label_vec, vector<string> &kernel_files)
{
	num_kernel_ = num_kernel;
	num_img_ = num_img;
	img_files_ = img_files;
	label_vec_ = label_vec;
	kernel_files_ = kernel_files;

	feature_ = Mat::zeros(num_img_, num_kernel_, CV_32FC1);
	vote_feature_ = Mat::zeros(num_kernel_, 1, CV_32FC1);
}

void CConvMaxPooling::ConvMaxPooling(int step_conv, bool sign_expand_kernel)
{
	for (int i_img = 0; i_img < num_img_; i_img++)
	{
		IplImage *img = cvLoadImage(img_files_[i_img].c_str(), CV_LOAD_IMAGE_COLOR);
		int width_img = img->width;
		int height_img = img->height;
		int step_img = img->widthStep;
		for (int i_kernel = 0; i_kernel < num_kernel_; i_kernel++)
		{
			int width_kernel, height_kernel, channel_kernel;
			CSamplingKernel kernel;
			Mat img_convolution;
			kernel.load(kernel_files_[i_kernel]);
			kernel.convolution(img, img_convolution, sign_expand_kernel, step_conv);
			
		}
	}
}

float CConvMaxPooling::train(bool sign_expand_kernel, int step_conv)
{
	float result = 0;
	ConvMaxPooling(step_conv, sign_expand_kernel);
	result = trainWithRandomForest();

	return result;
}