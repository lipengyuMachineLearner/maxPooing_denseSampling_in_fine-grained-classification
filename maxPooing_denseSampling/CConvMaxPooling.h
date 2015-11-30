#pragma once

#include "opencv.h "
#include "CSamplingKernel.h"
class CConvMaxPooling
{
private:
	int num_kernel_;
	int num_img_;

	vector<string> kernel_files_;
	vector<string> img_files_;
	vector<int> label_vec_;

	Mat feature_;
	Mat vote_feature_;
private:
	void ConvMaxPooling(int step_conv, bool sign_expand_kernel);
	float trainWithRandomForest() { return 0; }

public:
	CConvMaxPooling(int num_kernel, int num_img, vector<string> &img_files, vector<int> &label_vec, vector<string> &kernel_files);
	float train(bool sign_expand_kernel, int step_conv);
	
};