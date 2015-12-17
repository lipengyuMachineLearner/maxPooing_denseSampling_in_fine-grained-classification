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
	Mat label_;

	int num_target_;
	int *targetClass_;
	

	ofstream logfile_;
	static string cof_;
	
private:
	
	float trainWithRandomForest(float *num_eachClass, int dim, int targetClass);
	bool generateConvolutionImage(vector<Mat> &img, Mat &img_convolution, int width, int height);
	bool getLabel(int targetLabel, float *num_eachClass);
	bool getFeatureImportant(const CvDTreeNode  *root);

public:
	CConvMaxPooling(int num_kernel, int num_img, vector<string> &img_files, vector<int> &label_vec, vector<string> &kernel_files, string convolutionSavePath, int num_target, const int *targetClass);
	void ConvMaxPooling(int step_conv, bool sign_expand_kernel);
	float train();
	
};