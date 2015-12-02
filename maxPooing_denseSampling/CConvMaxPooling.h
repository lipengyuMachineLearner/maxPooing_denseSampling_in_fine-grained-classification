#pragma once

#include "opencv.h "
#include "CSamplingKernel.h"
#include <fstream>
#include <hash_map>
using namespace std;
class CConvMaxPooling
{
private:
	int num_kernel_;
	int num_img_;

	vector<string> kernel_files_;
	vector<string> img_files_;
	vector<int> label_vec_;
	string convolutionSavePath_;

	Mat feature_;
	Mat vote_feature_;
	hash_map<int, int> feature2kernel_;

	ofstream logfile_;
	static string cof_;
private:
	void ConvMaxPooling(int step_conv, bool sign_expand_kernel);
	float trainWithRandomForest() { return 0; }
	bool generateConvolutionImage(vector<Mat> &img, Mat &img_convolution, int width, int height);

public:
	CConvMaxPooling(int num_kernel, int num_img, vector<string> &img_files, vector<int> &label_vec, vector<string> &kernel_files, string convolutionSavePath);
	float train(bool sign_expand_kernel, int step_conv);
	
};